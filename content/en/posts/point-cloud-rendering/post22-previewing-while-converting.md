---
title: "Post 22: Previewing While Converting"
date: 2026-10-15T16:00:00+03:00
weight: 23
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - ux
---

Converting `toronto16` takes 140 to 160 seconds. A user who drops a 32 GB file and sees an empty window for two and a half minutes will assume the app has hung. The no-freeze rule ([Post 07](../post07-loading-100m-points-without-freezing/)) covers the UI; this post covers the *picture*: show something real within seconds, from the same pass that is already reading the data.

## The idea

The converter's second pass (counts) already streams every point. Let that pass also keep **every stride-th point**, up to the LOD point budget (30 M points by default), and hand those points to the UI in waves of 1 M. The UI loads each wave into the normal pool as an ordinary cloud (the "preview row"), so the user can see, rotate and zoom a uniform sample of the whole set while the real octree is built. When the LOD cloud appears, the preview row is removed.

```
count pass (converter thread):  every stride-th point  ->  waves of 1 M points (first wave: 64 k)
UI thread (preview_ingest):     wave -> pool range (sized once for the whole sample) -> upload tasks (Post 06)
conversion done:                LOD cloud appears -> preview row removed -> pool holds exactly the LOD region
```

The preview is drawn by everything the app already has: compute raster, colour modes, picking. One compromise is written in the code: the preview arrives in waves, so it is **not Morton-ordered**, and "the compute raster is somewhat slower on unordered points" ([Post 05](../post05-morton-order/) is why).

## Memory, once

The first wave creates the row and reserves the pool range for the *whole* sample, with at most one `pool_realloc` and the same VRAM check as a normal ingest ([Post 07](../post07-loading-100m-points-without-freezing/)). If the sample does not fit the VRAM budget, the preview is skipped and a line is printed; the conversion continues. Each wave is then copied into the arrays and queued for upload exactly like a loaded cloud. A bench check asserts the lifetime: the preview row exists before the LOD cloud, is removed after it, and the pool holds exactly the LOD cloud afterwards (63.9 M or 67.6 M points before, 37.8 M after).

## What it cost, measured

Whether the sampling slowed the conversion was a real question, so the benchmark compares `PCE_PREVIEW=0` (the sink is not installed) with the default, on a fresh conversion each time with the cache deleted:

| input | preview | first wave | complete | pass sum |
|---|---|---|---|---|
| 4 tiles, 78.3 M (5 runs) | off | - | - | mean 9.22 s |
| 4 tiles | on | 1.0-1.1 s | 2.3-2.5 s | mean 9.46 s (**+0.24 s, +2.6 %**) |
| toronto16, 1.25 B (2 runs) | off | - | - | 165.7 / 143.6 s |
| toronto16 | on | 18.2 / 17.6 s | 34.0 / 31.9 s | 156.2 / 146.7 s |

On the small set the preview costs the count pass about 0.3 s (+2.6 % overall). On toronto16 the difference is below the run-to-run noise: the indexing pass alone varies from 79 to 93 s between runs.

## The first number missed its target

The research notes set a goal: first point within **15 s** of the start on toronto16. The first implementation reached **17.6-18.2 s**: a miss by about 3 s, recorded as a miss. Of that, 12.8-13.3 s was the bounds pass, which has to finish before the count pass can start sampling.

A first fix attempt did nothing, and the write-up shows it: shrinking the first wave to 64 k points changed nothing on toronto16 (first wave 16.8-16.9 s). The measurement explained it in one sentence: **"the wave was ready, the converter was not."** The real cost was upstream. For a LAS file the converter decides how to shift 16-bit colours down to 8 bits by scanning the *whole* mapped file (all 1.25 billion records, about 4 s on 32 GB), once before the first batch of *every* pass, and the count pass paid it again.

The fix remembers the scan per `(path, size, mtime, layout)`. Result on toronto16:

| | bounds | first wave | complete | count pass | converted |
|---|---|---|---|---|---|
| before | 12.5 s | 16.8 / 16.9 s | 32.5 s | 19.9-20.0 s | 150.7-151.9 s |
| scan cache, preview on | 9.4 / 11.9 s | **9.7 / 12.2 s** | 25.6 / 28.4 s | 16.1 / 16.3 s | 138.7 / 143.2 s |
| scan cache, preview off | 12.2 s | - | - | 15.6 s | 142 s |

The first wave now comes 0.3 s after the bounds pass ends, and the target is met (9.7-12.2 s; the rest is the bounds pass itself, which varies with the OS file cache). It also made the **whole conversion 4-10 s faster**, since the distribute and index passes skip the scan too (count pass 20 s to 16 s). Preview on versus off, same build: 143 s against 142 s, within noise.

## An honest footnote

The `--bench-lod pick + remove` step fails on toronto16 with `PCE_LOD_VIEWS=1` ("picked 0 at -1.00 px from the centre, 0 points freed"). The write-up says it is **pre-existing**: the main checkout's build with no preview code fails the same way, on the existing `.pco` and on a fresh conversion. It is listed, not hidden, and not blamed on the new code without proof.

## Checklist

1. **Use the pass that already reads the data** to produce a preview; do not add a pass.
2. Make the preview a **normal pool row** so every feature works on it; accept an unordered-points penalty.
3. **Reserve memory once** and bail out gracefully if it does not fit.
4. **Measure with the feature off** in the same benchmark; report the cost as a percentage and as noise.
5. When the first fix does nothing, **find what the wave was waiting for**; here a 4-second scan, not the wave size.
6. Record **misses and pre-existing failures** next to the numbers.

Part 5 ends here. Part 6 is about editing, on the GPU: first, selection and picking. [Post 23](../post23-gpu-selection-and-picking/).

## Where this lives in the code

`src/app/lod/` (`LodJob`, preview waves), `src/app/loading/ingest.cpp` (`preview_ingest`), `src/io/octree.cpp` (the sampling sink, colour-scan cache), `PCE_PREVIEW`, `PCE_LOD_OUT`. Numbers: BENCH.md "R1.1" and "R1.1 follow-up". Commits 586e0a1, 626b7d2, e0b6069.
