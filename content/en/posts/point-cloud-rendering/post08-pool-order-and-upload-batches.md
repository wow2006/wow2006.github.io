---
title: "Post 08: Pool Order and Upload Batches, Debugging with Colour Modes"
date: 2026-10-14T17:00:00+03:00
weight: 9
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - debugging
---

Part 2 so far: the ring ([Post 06](../post06-persistent-mapped-staging-ring/)) and the loaders ([Post 07](../post07-loading-100m-points-without-freezing/)). This short post is about something that sounds minor and saves hours: **debug views that show how the pool was filled.** The editor has two, "Order block" and "Upload batch", and they exist because the first time anyone looked at a loaded cloud with a batch-coloured pool, it looked broken.

## What a point's pool index is

By now a point's pool index has an odd meaning (Posts 03 and 05):

1. Every loaded cloud was Morton-sorted, cut into full 128-point blocks, and the **blocks were shuffled globally** with a fixed seed. The trailing partial block stays last.
2. The cloud was then uploaded in pieces of `kUploadChunkPoints` points (5 592 405, i.e. 64 MB / 12 B), all attributes of one piece, then the next.

So the pool index of a loaded cloud is not a spatial coordinate. It is "which random 128-point block, then which point inside it". Inside one block the points are spatially close; neighbouring blocks in the pool are unrelated.

## Upload batches

A **batch** is one upload piece, a contiguous range of pool indices. For a streamed `.pco` cloud (Post 18) a batch is one cache page of `kLodPage = 4096` points. The renderer builds a sorted table of batch start indices on the CPU:

```cpp
struct BatchSpan { std::size_t base, count, size; };
inline std::vector<uint32_t> batch_starts(std::span<const BatchSpan> spans)
{
    std::vector<BatchSpan> s(spans.begin(), spans.end());
    std::ranges::sort(s, {}, &BatchSpan::base);
    std::vector<uint32_t> starts;
    for (const BatchSpan& c : s)
        for (std::size_t i = 0; c.size && i < c.count; i += c.size)
            starts.push_back(static_cast<uint32_t>(c.base + i));
    return starts;
}
```

and the shader finds a point's batch with a binary search over that table (`batch_index` is the CPU twin, same loop, unit-testable without GL):

```glsl
vec3 batch_color(uint idx) {
    if (nBatches == 0) return vec3(0.5);
    int lo = 0, hi = nBatches;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (batchStart[mid] <= idx) lo = mid; else hi = mid;
    }
    return hue_color(fract(float(lo) * 0.61803399));   // golden-ratio hue step keeps neighbours apart
}
```

The table lives in an SSBO (binding 14) that is bound and updated only while the mode is on, so the debug view costs nothing otherwise. The same `point_color()` function serves both renderers (GL_POINTS vertex shader and the compute resolve), so the debug colouring works everywhere with no extra code.

## "Why does my loaded cloud look like confetti?"

Toronto-3D L001 has about 20 M points, so it uploads as about four batches, four colours. But because the 128-point blocks were shuffled over the *whole* cloud, each batch is a random quarter of all blocks. Every batch therefore covers the entire scene, and you see 128-point patches of four colours interleaved at a fine scale, following the scan-line streaks of the data.

That is the expected picture, not a broken upload. It shows that batch boundaries carry no spatial meaning for loaded clouds. The repository documents it next to the colour mode, so the next person who sees confetti finds the answer.

For a streamed cloud the same mode shows 4096-point pages. Pages are handed to octree nodes as they stream in and are evicted, so neighbouring pages belong to unrelated nodes: also fragmented, for a different reason.

## Order block

The second mode colours each point by `hash(pool_index / 128)`, one colour per shuffled block:

```glsl
vec3 block_color(uint idx) {
    return hue_color(float((idx >> 7) * 2654435761u >> 8) * (1.0 / 16777216.0));
}
```

Each patch you see is one Morton-compact 128-point block, and the patch size does not change with the cloud size. The hash is a multiplicative hash of the block number, not the golden-ratio step used for batches, because float precision runs out when block numbers reach the tens of thousands (the hue step stops separating neighbours). For a LOD cloud the pages are multiples of 128 points, so blocks remain the right unit.

Together the two answer different questions:

| question | mode |
|---|---|
| did the block shuffle and sort work? are the blocks tight? | Order block |
| how was the pool filled over time? did every piece land? | Upload batch |
| where are the octree cells? | neither: View panel, Octree LOD, "Show octree levels" (streamed clouds) |

## Progressive display is correct by construction

One more property the batch view makes visible: a big cloud appears in steps, because the scene counts a point as drawable only when **all** its attributes have been copied:

```cpp
std::size_t uploaded(const SceneCloud& e)
{
    std::size_t n = e.count;
    for (int a = 0; a < kAttrs; ++a)
        n = std::min(n, e.bytes[a] / kAttrBytes[a]);
    return n;
}
```

The raster, the resolve ranges and the draw calls all use `uploaded(e)`, not `count`, so they never read pool memory the ring has not filled yet. With an attribute-by-attribute upload that rule is the difference between a clean progressive load and flashes of garbage colour.

## Name things for the GPU tools too

The same "make the invisible visible" idea applies to captures. The renderer wraps passes in `DebugGroup` (a `glPushDebugGroup` range) and labels every buffer, texture, program and framebuffer with `glObjectLabel`, e.g. `pool pos`, `raster fb64`, `staging ring`, `compute raster`, `resolve`. In Nsight Graphics or RenderDoc the capture then reads as a tree of named passes and named resources instead of numeric ids. It cost one RAII struct and a few lines per resource.

## Take-aways

1. If your pool order is not the file order, build the colour mode that shows it before you need it. It is a hash and a binary search.
2. Write the CPU twin of every shader lookup (`batch_index`, `range_index`) in a GL-free header and test it. The shader and the test then agree by construction.
3. Document "this is expected" pictures next to the debug mode. The pool-order notes in the repo exist because the confetti looks like a bug.
4. Only draw what has fully arrived: track uploaded points as the minimum over attributes.
5. Label GPU objects and passes from day one.

That closes Part 2. Part 3 is the core of the series: replacing the draw call with a compute rasterizer. [Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/) starts with the 64-bit atomic.

## Where this lives in the code

`docs/pool-order-and-batches.md`, `src/render_model/render_model.hpp` (`BatchSpan`, `batch_starts`, `batch_index`), `src/render/render.cpp` (`batch_color`, `block_color`, `point_color`, `DebugGroup`, `gl_label`), `src/app/scene/scene_cloud.cpp` (`uploaded`). Commits: 37788be (octree level overlay, Order block mode, pool order doc), dd3e4db (Nsight debug groups and labels, Upload batch mode).
