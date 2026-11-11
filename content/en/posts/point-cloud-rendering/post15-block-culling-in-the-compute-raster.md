---
title: "Post 15: Block Culling in the Compute Raster"
date: 2026-10-15T09:00:00+03:00
weight: 16
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - culling
---

[Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/) ended with a puzzle: the compute raster costs the same at close-up (1.097 ms) as at fit-all (1.068 ms), even though a close-up view throws away almost every point. The kernel still reads and transforms every position, and reading 940 MB is the bill. This post fixes that with the 128-point blocks from [Post 05](../post05-morton-order/): a workgroup looks at its block's bounding box first and leaves without reading a single point if the box is outside the view.

## Setup: one box per block

After the Morton sort and block shuffle, block `b` of a cloud is points `[128 b, 128 b + 128)` and is spatially tight. `block_bounds` computes an axis-aligned box for each block, once, on the loader thread:

```cpp
struct BlockBox { glm::vec3 lo, hi; };        // 24 B
static_assert(sizeof(BlockBox) == 24);
std::vector<BlockBox> block_bounds(std::span<const glm::vec3> pos, std::size_t max_threads = 0);
```

For 78 M points that is 15 MB, in a per-cloud SSBO (`float[6]` per block, `std430`). The comment in the header makes a point worth repeating: the raster reads every box even where nothing is culled, so at the fit view the boxes are extra bandwidth, which is why they are 24 bytes and not two `vec4`s. NaN coordinates are ignored; a block with no finite point gets an inverted box (`lo = +inf`, `hi = -inf`).

## The cull test in the raster

The raster is compiled twice: `PCE_CULL 0` is the kernel from Post 09, and `PCE_CULL 1` makes **one workgroup equal to one 128-point block**. Eight threads of the workgroup each test one corner of the block's box, then a barrier, then the whole workgroup decides:

```glsl
shared uint corners[8];
if (gl_LocalInvocationID.x < 8u) {
    const uint k = gl_LocalInvocationID.x;
    corners[k] = 0u;
    if (block0 != 0xffffffffu) {        // an unaligned chunk: no box read at all
        const uint b = 6u * (block0 + gl_WorkGroupID.x);
        const vec3 lo = vec3(blocks[b], blocks[b+1u], blocks[b+2u]), hi = vec3(blocks[b+3u], blocks[b+4u], blocks[b+5u]);
        corners[k] = cornerPlanes(mvp, vec3((k&1u)!=0u ? hi.x : lo.x, (k&2u)!=0u ? hi.y : lo.y, (k&4u)!=0u ? hi.z : lo.z));
    }
}
memoryBarrierShared();
barrier();
// All 8 corners outside the same plane: nothing of the block can pass the clip test below.
if ((corners[0] & corners[1] & ... & corners[7]) != 0u)
    return;
```

`cornerPlanes` returns a 6-bit mask, one bit per clip plane (`x > w`, `x < -w`, `y > w`, `y < -w`, `z > w`, `z < 0`), set when the corner is **outside that plane**. AND the masks of the eight corners: a bit survives only if all eight corners are outside the *same* plane, which proves the whole box (and so every point in it) is outside. That is the standard conservative frustum test, and it is linear in clip space, so it needs no division by `w` and no special case for corners behind the eye.

Two details are easy to get wrong:

- **The barrier comes before any `return`.** All threads of the workgroup must reach it, so the early exit is after it.
- **The test is conservative by a rounding margin.** A corner counts as outside a plane only if it violates it by more than `1e-5 * |mvp| * |p|`, the rounding error of the matrix product (`kBlockCullTol`). The points inside the box are transformed with the same matrix but not the same float operations, and a point the raster would have kept must never lose its block. A CPU twin, `block_corner_planes`, lives in the GL-free header, with a test for a box sitting exactly on a clip plane: 132 cases where a margin-free test would drop a drawn point.

## Aligning the chunks

Culling assumes workgroup `g` of a dispatch is block `block0 + g`. That holds only if the dispatch starts on a block boundary. Post 03's chunking splits a cloud at arbitrary point counts when it exceeds the SSBO limit, so there is a variant, `ssbo_chunk_blocks`:

```cpp
SsboChunk c = ssbo_chunk(first, count, stride, align, max_bytes);
if (c.count < count && c.count >= unit) {      // a chunk that does not end the range
    c.count -= c.count % unit;                  // holds a multiple of 128 points
    c.size = c.skip * 4 + c.count * stride;
}
```

so every chunk except the last starts and ends on a block boundary. A binding too small to hold one block stays unaligned and passes `block0 = ~0u`, which disables the box read for that chunk. The test `PCE_MAX_BLOCK=300000000` forces two chunks on a 39.7 M-point cloud so the aligned path runs on small data.

## Don't cull what can't be culled

If the whole cloud's bounding box is inside the view, no block can be culled, and the extra box reads are pure cost. The compute raster does not check for that: it runs the cull variant whenever culling is on, and the fit-view row below shows the price. The mesh path of [Post 16](../post16-mesh-shaders-for-points/) does check, with `box_inside(mvp, cloud box)`, because its task shader cost about 0.65 ms at fit-all when it culled nothing. The same test in front of the compute raster is the obvious next step; it has not been measured there. For A/B runs, `PCE_NO_CULL=1` (or the Stats panel checkbox) switches culling off.

## Results

Toronto, 78.3 M points, 2400 x 1350, compute renderer, 600 frames, p50, back to back with the old build (the second value in each cell is a second run):

| path | raster before | raster with culling | change |
|---|---|---|---|
| close | 1.100 / 1.099 ms | **0.344 / 0.344 ms** | **3.20x faster** |
| fit | 1.073 / 1.073 ms | 1.091 / 1.092 ms | +1.7 % slower |
| fly | 1.089 / 1.086 ms | 0.954 / 0.940 ms | 1.15x faster |

The GPU scene time at close-up falls from 1.238 to 0.446 ms. At the fit view every block is visible and nothing is culled, so the only effect is the box reads and the barrier: +1.7 %, paid because the raster has no whole-cloud skip.

The project wrote its acceptance criteria *before* measuring: close view at least 2x faster, fit view no more than 2 % slower, and a kill condition (under 1.3x close-up means revert). Result: 3.2x, +1.7 %, "just inside". That is a good habit: decide what counts as success while you are still unattached to the code.

## Proving it draws nothing different

Culling may drop only points that would draw nothing. Two checks:

1. **Image diff, culled vs unculled** at fit, close and ortho (the `--bench` harness at 1600 x 900, size 3): **0 of 1 440 000 pixels** differ in all three, also with a forced two-chunk cloud.
2. **Mutants** (Post 11): any-corner instead of all-corners gives 130 221 differing pixels; a block index off by one gives 655 116. The harness catches both. Dropping the rounding margin gives 0 on these views, so the CPU test above guards it.

## Why this is only possible with the block order

Culling by **index range** would be useless: a cloud's pool indices are a random mix of blocks (Post 05, 08). It works at the 128-point block level because each block is spatially tight, and because each workgroup is one block. Without the shuffle's internal coherence the boxes would be large and almost always visible.

It also does not help the fit view, and it cannot: if all 78 M points are on screen, all must be read. Going lower there needs a coarser representation, which is the job of the LOD octree in Part 5.

## Checklist

1. Choose a **block size that equals your workgroup size**, and sort so blocks are spatially tight.
2. Precompute a **24-byte box per block**; keep the CPU twin of the plane test.
3. **8 threads, 8 corners, AND the plane masks**, barrier before any return.
4. Add a **rounding margin** and a unit test that sits exactly on a plane.
5. Align chunks to blocks, and have an escape (`~0u`) for those that cannot be.
6. Skip culling when the whole cloud is in view (the mesh path does; the compute raster pays 1.7 % for not doing it); keep a switch for A/B.
7. Define success criteria first; verify with an image diff and with mutants.

Next: the same blocks, culled by a task shader, in a very different pipeline: [Post 16](../post16-mesh-shaders-for-points/).

## Where this lives in the code

`src/core/order.hpp` (`BlockBox`, `block_bounds`), `src/render/render.cpp` (`kRaster` with `PCE_CULL`, `PCE_CORNER_GLSL`, `raster_points`), `src/render_model/render_model.hpp` (`ssbo_chunk_blocks`, `block_corner_planes`, `kBlockCullTol`), tests in `tests/render_model/test_render_model.cpp`. Numbers: BENCH.md "R3 - block culling". Commit 8b266d1.
