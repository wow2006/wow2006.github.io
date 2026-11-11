---
title: "Post 26: Edits on Streamed Clouds"
date: 2026-10-15T20:00:00+03:00
weight: 27
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - lod
  - editing
---

A streamed cloud ([Post 18](../post18-out-of-core-octree/)) keeps only a few percent of its points in VRAM at any moment. Select a region at a coarse zoom level and press Delete, and the question is: *which points?* The ones you can see, or every point in that region at every level of detail, including nodes that are not loaded? The answer decides everything about the design, and the notes record that it was a deliberate decision, written down before the code.

## Start by checking the premise

The fix list opened with a P0 item: "node flags are cleared to 0 on (re)load; select/hide/delete on a streamed cloud is lost when a node is evicted". The first step was to *verify* that. The finding: **nothing was being lost, because nothing was ever edited.** Every edit op, Save, Merge, Subsample, ICP and Segment simply skipped LOD clouds (`!c.lod` in `run_op`; the roadmap said "view only"). So the real task was to make LOD clouds editable, not to fix a bug. That distinction changed the plan from a patch to a design.

## The semantics, decided

Two options were written out:

- **(a) Edit what you see**: an edit applies only to resident points.
- **(b) Edit the region**: an edit applies to every point inside the selection region, including points in nodes not loaded yet.

Decision: **(b) for every op**, selection included. The reason is data integrity: with (a), a saved file would still contain points the user thought were deleted. The consequences of (b) shape everything:

1. The **CPU holds the truth**, not the GPU. Per node there is a store of flags and classes, and per cloud there is an **op log**.
2. An op **runs immediately on the resident nodes** and is **replayed** by every other node when it loads, *before* its upload is issued, so a node never shows unedited.
3. The GPU pages are a **mirror**.

```cpp
struct LodOp {                       // one logged edit op
    int op; uint32_t arg;
    glm::mat4 mvp;                   // octree frame -> clip, the cloud's mvp when it ran (transform included)
    glm::ivec2 size;                 // framebuffer (select)
    std::shared_ptr<const std::vector<uint32_t>> mask;   // polygon_mask bits (Post 23), shared by the clouds it ran on
    glm::ivec4 box;                  // pixels that can be set in the mask (empty: none)
    ClassMask hidden;                // the classes hidden when it ran
    std::unordered_map<uint32_t, std::vector<uint32_t>> deltas;   // undo data per node (Post 25's layout)
};
struct LodStore {                    // a node's edited state
    std::unique_ptr<uint8_t[]> flags, cls;   // null: all flags 0 / the file's classes
    uint32_t applied = 0;                    // ops[0, applied) are in it
    uint32_t version = 0;
};
```

A node's flags and classes are allocated on the **first op that changes them** (1 byte per point each), so an unedited node costs nothing. `applied` says which prefix of the log the node has seen; loading a node replays `ops[applied..]`.

The op log stores what a GPU `kEdit` would need (matrix, mask, framebuffer size) because the replay is the *same projection, clip test and pixel* as the GPU pass ([Post 23](../post23-gpu-selection-and-picking/)), executed on the CPU. A node holds at most 20 000 points, so a CPU replay is cheap, and the bounding `box` of the mask lets the replay cull a whole node by projecting its cube: a select against 5.8 M resident points of `toronto16` took 9 ms plus 10 ms to mirror to the GPU. The choice of CPU replay over a GPU pass for non-resident nodes is a recorded trade-off ("nodes are <= 20 k points").

### Visible-only is the exception

A "visible-only" select needs the framebuffer ID. A resident point counts as visible when it owns its pixel in the frame's `fb64`. A node that was *not* resident when the user selected was not drawn, so a replay of a visible-only select **hits nothing**. That matches what the user saw, and the rest of option (b) still holds for ordinary region ops.

## What the GPU sees

Pages mirror the stores: flags and classes are uploaded with the node (copied into a buffer the upload tasks point at), re-uploaded when an op, undo or redo changes a resident node (the `mirror` list), and **zeroed when an edited node is evicted** (`clear_pages`) so free pages always have flags 0. The raster and the resolve read flags exactly as for a loaded cloud ([Post 24](../post24-flags-and-class-edits-as-compute-passes/)).

Counts (selected, hidden, deleted) cover the *loaded* points of LOD clouds; the tooltip says so.

## Why no tree rebuild

Editing a multi-resolution tree usually means rebuilding the coarse levels from the survivors. Not here, because of a property proven in [Post 18](../post18-out-of-core-octree/): **every point is stored once, in one node**. A deleted point disappears at every level. The design notes recorded this as "investigate first, then skip F1.6": the answer was written into `octree.hpp` and `check_octree` tests "every point once at its position".

The price: a coarse level shows fewer points of a region after a delete, until the user zooms in, since levels are not re-sampled from the survivors. The project measured how large that effect is:

## Density holes after deletes

A **hole** is a deleted sample at an inner level whose grid cell still holds a live point in a descendant. It is a place where the coarse view lacks a point that full detail has. On the 4-tile `.pco` (78.3 M points, 14 823 nodes, levels 0-13), holes as a percentage of each level's samples:

| deleted set | levels 0-6 | all inner levels |
|---|---|---|
| (a) 1 % random | 0.78 % | 0.62 % |
| (b) class 3, natural (8.4 % of points) | 0.25 % | 0.055 % |
| (c) class 7, cars (5.9 %) | 0.18 % | 0.049 % |
| (d) a box covering 58.8 % of points | 0.02 % | 0.004 % |

For random deletion the hole share simply equals the delete share (0.6 to 1 % of samples), spread evenly. For concentrated deletions the holes pile up in the first levels, where a cell is 4-8 m wide and mixes classes: class 3 at level 0 is 11.6 % of 4 840 samples. At most levels it is a fraction of a percent, which is why no re-sampling pass was built. The measurement exists so that decision can be revisited with evidence.

## Undo on a streamed cloud

Deltas of [Post 25](../post25-undo-redo-as-gpu-xor-deltas/) are over pool positions, which are invalid once LOD pages move. So for LOD clouds each op records, **per node it changed**, the XOR delta in the same layout, and keeps them in RAM, capped at 1 GB for all LOD clouds (oldest first; the op itself stays in the log). Undo and redo apply the deltas to the node stores and mirror resident nodes. There is one global history: an `Edit` lists the LOD clouds whose log took the op (`Edit::lod`).

The tests cover: edit, evict, undo, reload; undo and redo across evictions and during uploads; the history cap; and a coarse box delete reaching all 153 943 points of the box (1 040 resident), plus 300 tiny boxes equal to the direct hit test's union (which catches over-eager culling).

## Saving bakes the log

Exporting must apply every logged op to every node, not just the resident ones. `lod_save_source` streams each node through the op log on a **snapshot** of the stores (`lod_bake`), so editing and streaming continue meanwhile, into the LAS/LAZ/PLY writers, skipping deleted points and applying classes. End-to-end on the 4-tile `.pco` (78.3 M): box-delete at a coarse view, save, reload: **66 974 716 points written, 11 345 494 deleted, equal to every tree point inside the box at full depth** (10.1 M of them loaded at the time); one reloaded point lands in the box after LAS's 0.1 mm rounding; 2.8 s, 1.74 GB. The full 1.25 B-point set was not saved in the bench (estimated 45 s and 33 GB).

Not done: a sidecar file so a reopened `.pco` restores edits without re-export.

## Checklist

1. **Verify the bug report first**; it may be a missing feature.
2. **Write the edit semantics down** before coding, and pick the one that matches what the saved file must contain.
3. Keep the truth where the **complete data** is (CPU stores plus an op log); make the GPU a mirror.
4. **Replay logged ops on load**, before the upload is issued.
5. Rely on **each point stored once** to avoid rebuilding coarse levels, and measure the visual cost.
6. Key undo by node, in RAM, with a cap.
7. Prove it end to end: **delete at a coarse level, save, reload, check the full-depth tree**.

Next: an operation with a very different shape, whole-cloud neighbour search, that needed time-slicing to avoid freezing the GPU: [Post 27](../post27-knn-outlier-removal-in-time-sized-slices/).

## Where this lives in the code

`src/lod/lod.hpp`, `lod.cpp` (`LodOp`, `LodStore`, `lod_edit`, `lod_undo`, `lod_snapshot`, `lod_bake`), `src/lod/holes.cpp` (`lod_density_holes`), `tools/lodholes.cpp`, `src/app/lod/` (mirroring). Numbers: FIXES F1, BENCH.md "R8". Tests: `check_lod_edits`.
