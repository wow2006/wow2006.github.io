---
title: "Post 30: Putting It Together, a 235M-Point Frame Budget"
date: 2026-10-16T09:00:00+03:00
weight: 31
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - performance
---

Twenty-nine posts of pieces. This one assembles them into a budget: for a scene of **235 million points**, where does the memory go, where does the frame time go, what limits are close, and what remains unmeasured. Every figure below is quoted from the project's notes or derived by simple arithmetic from them, and derived figures are marked.

## The scene

`Toronto 4 tiles x 3` copies = **234 960 630 points**, loaded as three clouds into one pool of about 224 Mi points (224 x 2^20). Before the fix of the resolve's chunking ([Post 03](../post03-one-big-pool/)), this scene could not use the compute rasterizer at all in elevation mode, because positions needed one 2.69 GB binding, more than the 2 GiB limit.

## The memory budget

All attributes are resident ([Post 13](../post13-colour-modes-without-re-upload/)), at 20 bytes per point ([Post 02](../post02-data-layout-for-gpu-point-clouds/)):

| what | size | source |
|---|---|---|
| positions, 12 B/pt | ≈ 2.8 GB (derived: 235 M x 12) | two SSBO chunks of 178 956 960 points |
| colour, intensity, class, flags (4 + 2 + 1 + 1 B) | ≈ 1.9 GB (derived) | |
| block boxes for culling, 24 B per 128 points | ≈ 44 MB (derived) | [Post 15](../post15-block-culling-in-the-compute-raster/) |
| `fb64`, 8 B/pixel | 26 MB at 2400 x 1350 (derived) | [Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/) |
| selection mask, 1 bit/pixel | 405 KB | [Post 23](../post23-gpu-selection-and-picking/) |
| staging ring | 192 MB | [Post 06](../post06-persistent-mapped-staging-ring/) |
| undo history | capped at one eighth of VRAM: 3 GB on the 24 GB card | [Post 25](../post25-undo-redo-as-gpu-xor-deltas/) |
| one flag plane delta | n/8 B ≈ 29 MB (derived); a class delta ≈ 235 MB (derived) | |

About 4.7 GB for the cloud itself, in a 24 GB card. The scene is far from the VRAM limit and not far from the *binding* limits: the compute path supports **715 M points** (four resolve chunks of 178.96 M), after which `GL_POINTS` takes over and the Stats panel says why ([Post 17](../post17-choosing-a-renderer-at-runtime/)). The index field of the 64-bit key tops out at about 4.29 billion.

## The frame budget

For the 78.3 M-point scene at 2400 x 1350 (the most carefully measured one), a fit-all frame costs:

| stage | ms | share |
|---|---|---|
| clear `fb64` (26 MB) + raster of 78.3 M points | 1.068 | 86 % of scene |
| resolve (3.24 M pixels) | 0.036 | 3 % |
| rest of the scene pass (blit etc.) | ≈ 0.14 (derived: total minus the two above); EDL adds 0.003-0.005 | |
| **GPU scene total** | **1.241** | |

Variations, measured: colour mode elevation +0.54 ms, intensity +0.15, class +0.08 ([Post 13](../post13-colour-modes-without-re-upload/)); point size 7 +0.12 ([Post 12](../post12-point-size-without-sprites/)); flags set +0.13; close-up with block culling 0.446 ms total ([Post 15](../post15-block-culling-in-the-compute-raster/)).

At 60 frames per second there are 16.7 ms. The scene takes 1.24 of them, which leaves about 15 ms for everything else: the UI, uploads, the edit ops that are issued that frame, the LOD selection (at most 0.06 ms), and picking.

Scaling to the 235 M-point scene (benchmarked in the 1600 x 900 `--bench` window, size 1, so not directly comparable to the 2400 x 1350 numbers above):

| view | `GL_POINTS` | compute | image diff vs `GL_POINTS` |
|---|---|---|---|
| fit all | 114.52 ms | **3.06 ms** | 770 of 1 440 000 px (35 coverage) |
| street 15 m | 7.02 ms | 3.09 ms | 1 500 of 1 440 000 px (1 269 coverage) |

114.5 ms is 9 frames per second, and it is what the elevation mode would have been before the chunking work. The compute path takes 3.06 ms. The same diff pixel counts appear at 78.3 M points with one binding and with forced 3-chunk bindings: **the chunking changes no pixel**. The chunked shader cost at 78.3 M points (elevation, fit / street) is 1.134 / 1.165 ms with one chunk against 1.138 / 1.166 ms with three: the indirection is free at that size.

Rough scaling, an inference not a measurement: the raster reads every position once, so raster time grows linearly with points, and the pixel-bound stages do not grow at all. The like-for-like pair is in the same BENCH section: elevation mode, fit-all, the same 1600 x 900 window, 1.134 ms at 78.3 M against 3.06 ms at 235 M, about 2.7x for 3x the points, close to linear. (Setting the 1.24 ms RGB figure at 2400 x 1350 against 3.06 ms would mix two windows and two colour modes.)

## When the cloud is bigger than the card

`toronto16` has 1.25 B points: 15.1 GB on disk as `.pco` v2, 25 GB in pool layout. It is drawn from the LOD cache, with a budget of 30 M points (720 MB of VRAM), in a measured **0.115-0.155 ms** of GPU scene time, with 0.27 to 3.7 M points actually drawn ([Post 18](../post18-out-of-core-octree/)). Cold from disk, full detail takes 0.04-0.07 s after prefetch and parallel fill ([Post 20](../post20-streaming-from-a-cold-disk/)). Converting takes 144 s, with a preview within 10-12 s ([Post 22](../post22-previewing-while-converting/)).

The shape of the whole design is: **make the cost proportional to pixels, not to points**. Rasterization with an atomic gives one resolve per pixel. Block culling skips reads for what is off screen. The octree does it for what is too fine to see. Each step is the same idea at a different level.

## What costs time that is not drawing

From the no-freeze audit ([Post 27](../post27-knn-outlier-removal-in-time-sized-slices/)), the remaining synchronous costs, deliberately left as documented exceptions:

| operation | cost | size |
|---|---|---|
| pool compaction (Remove, merge swap) | 50-480 ms | 10-290 M pool |
| pool growth | 7-226 ms | 10-290 M |
| `snapshot_edits` (flag + class readback before a save, merge, ICP...) | 0.45 ms per M points; 109 ms | 143 M |
| edit ops | box / lasso select 47-53 ms, every other op ≤ 12 ms | 78.6 M, loaded Linux machine |

Everything user-triggered above 100 ms is a job with progress and Cancel; GPU work is cut so no frame passes 50 ms: the measured worst frame during the largest outlier job was 44.6 ms.

## What is not measured

Honesty about the edges, from the notes:

- **No 10 M and 500 M rows**, and one GPU model for most numbers. The roadmap's task 3.5 is open.
- The reworked **mesh-shader numbers come from a laptop GPU**; I found no re-run on the 4090.
- The **Morton bit count** result (10 bits) rests on medians the notes call suspicious; an adaptive choice for km-scale files is not done.
- **Elevation costs 0.54 ms** more than RGB at fit-all; both fixes (unproject z, or keep a z copy) are written down and unbuilt.
- The **raster's near-bandwidth-bound** claim (≈ 880 GB/s of positions against ≈ 1 TB/s peak) is arithmetic from timings, not a profiler counter.
- **Linux drivers** show a 35x slowdown on one kernel that nobody has explained.
- `GL_NV_mesh_shader`, 64-bit atomics: **NVIDIA only**. There is no portable fallback beyond `GL_POINTS`.
- A sidecar for LOD edits, even-sized and round points, LOD clouds in merge / ICP / segmentation: not built.

## The design in ten sentences

1. Store a cloud as **SoA in one arena**, with a double origin and float offsets.
2. Keep one **pool of five buffers** with global indices, bound in windows.
3. Build the **camera-relative matrix in double**, cast late; use reverse-Z with a float depth.
4. **Sort by Morton code and shuffle 128-point blocks**; the block is the unit of work.
5. Move bytes through a **persistent-mapped ring** with fences, off the render thread.
6. **Rasterize with a 64-bit `atomicMin`**, resolve once per pixel, write real depth.
7. Cull whole blocks by box; optionally dedupe in a mesh shader.
8. Stream an **octree of real points** through a page table into the same pool.
9. Edit with **compute passes over a byte per point**, and undo with **XOR deltas** in VRAM.
10. **Measure everything**, on a pinned path, against a reference image, and log what failed.

## Where to go next

If you build this yourself, the order that paid off most: layout and the pool (Posts 02-03), the ring (06), the atomic raster and resolve (09-10), and the parity harness (11) before anything else, then culling (15), then LOD (18-21) when your data outgrows VRAM, and editing (23-27) last. Keep a ledger from day one (Post 28), and open a capture before you believe a story about bandwidth (Post 29).

## Where this lives in the code

The repository's `docs/` hold the design log: `ROADMAP.md` (tasks and done-when checks), `FIXES.md` (the fix plan), `BENCH.md` (every measured row), `rendering-engine.md` (the architecture overview), `pool-order-and-batches.md`. Numbers in this post: BENCH.md "F4.4", "F0.1", "F3", "Task 2.3", ROADMAP 3.2 and 6.1.
