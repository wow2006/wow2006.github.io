---
title: "Post 21: Feeding the Same Shaders from a Streamed Page Table"
date: 2026-10-15T15:00:00+03:00
weight: 22
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - streaming
---

[Post 18](../post18-out-of-core-octree/) chose which nodes to draw and [Post 20](../post20-streaming-from-a-cold-disk/) got their bytes off the disk. This post is the glue between them and the renderer, and its most important property is negative: **the rasterizer, the resolve, picking and the colour modes did not change**. A streamed cloud is drawn by the same code as a loaded one, because it lives in the same pool.

## A region of pages in the pool

A LOD cloud gets a fixed **region** of the pool, divided into pages of 4096 points. `4096` is a multiple of the raster's 128-thread workgroup, so a page is exactly 32 workgroups. The region's size is the cache budget times 1.25 plus 64 pages, capped at what the octree needs and at one position binding. A node takes `ceil(count / 4096)` pages, and its last page is partly empty (that tail is unused; its flags stay 0).

```cpp
struct LodCache {
    std::unique_ptr<Octree> tree;
    std::size_t base = 0;            // the region's first point in the pool (moves with compaction)
    std::size_t pages = 0;           // region size in pages
    enum State : uint8_t { kAbsent, kLoading, kResident };
    struct Node { State state; uint32_t pending; uint64_t drawn, selected; std::vector<uint32_t> pages; ... };
    std::vector<Node> nodes;         // per octree node
    std::vector<uint32_t> free;      // free pages
    std::vector<uint32_t> page_node; // node holding each page (~0: free)
    std::vector<glm::uvec2> draw;    // drawable pages: {first point relative to base, points}
};
```

The cache is plain C++ with no GL (`src/lod`), which is why it can be tested with a CPU stand-in for the pool. All the pool-index logic from [Post 03](../post03-one-big-pool/) applies unchanged: page `p` covers pool points `[base + 4096 p, base + 4096 (p + 1))`, in every attribute buffer.

## One frame of the cache

`lod_frame(cache, selected, frame, max_bytes, tasks)` does the whole job:

1. Mark the selected nodes with this frame's number.
2. For each selected node **in priority order** that is absent: take pages from the free list, **evicting** resident nodes when needed, and issue an upload task per page and attribute (position, colour, intensity, class). Stop at the byte budget for new uploads (what is left of the 192 MB allowed in flight, `kLodInflight` minus what is still pending) or when pages run out. "Keep the priority order: nothing behind it goes first."
3. Ask the OS to prefetch the new nodes' bytes (Post 20).
4. Rebuild `draw`: the pages that are **selected and resident**, as `{first, count}` pairs relative to `base`.

**Eviction** takes resident nodes that were neither selected this frame nor drawn in the last `kLodKeepFrames = 4` frames, least recently drawn first. The 4-frame grace period exists because a pick reads the framebuffer back a few frames after drawing it: evict and reuse the page immediately and the pick would name a point from a different node. (The notes admit a pick can still name a page reassigned since its frame; the 4-frame keep is the only guard.) When an edited node is evicted, its pages' flags are zeroed on the GPU so a free page never carries stale selection bits.

The tests audit this over hundreds of frames: per-frame budget, eviction rules, page ownership and counters, with tiny regions of 11 pages and late shuffled uploads; two views turning the cache over, and a returning view reloading only what was evicted. They were mutation-tested (eviction of selected nodes makes it thrash and never settle; the test catches that).

## The page table in the raster

The compute raster has a compile-time variant for pages, `PCE_PAGES`. The page list goes into an SSBO of `uvec2 {first, count}` and **one dispatch covers all pages**:

```glsl
#if PCE_PAGES
layout(std430, binding = 3) restrict readonly buffer Pages { uvec2 pages[]; };
#endif
void main() {
#if PCE_PAGES
    const uvec2 pg = pages[gl_WorkGroupID.x / PCE_PAGE_WGS];                         // PCE_PAGE_WGS = 4096 / 128
    const uint k = (gl_WorkGroupID.x % PCE_PAGE_WGS) * 128u + gl_LocalInvocationID.x;
    if (k >= pg.y) return;                                                           // past the page's used points
    const uint i = pg.x + k;                                                         // index relative to `base`
#else
    const uint i = gl_GlobalInvocationID.x;
#endif
    ...   // the same clip test, projection and atomicMin as Post 09; the key still carries base + i
}
```

```cpp
glDispatchCompute(entries * (page / 128), 1, 1);   // raster_pages
```

Everything after `i` is the code from [Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/): the global index goes into the 64-bit key, so **the resolve** finds colour with `rgba[index]` exactly as before; **picking** returns a pool index which `lod_point` maps back to `(node, index in node)` through `page_node`; **colour modes**, tint, EDL and the grid work untouched. The `CloudRange` table ([Post 10](../post10-the-resolve-pass/)) lists the region as one range with `uploaded(e)` unchanged.

What does differ is only the dispatch and the fallbacks:

| renderer | how it draws a LOD cloud |
|---|---|
| compute | `raster_pages`: one dispatch over the page table (`PCE_PAGES` variant, plus a flags variant) |
| `GL_POINTS` | `glMultiDrawArrays` over the `draw` list |
| mesh | `draw_mesh_pages` with a task-free mesh program that reads the same page table (`PCE_MESH_PAGES`) |

`raster_pages` returns false if the region's positions do not fit one SSBO binding (`c.count < points`), and the caller then falls back to the multi-draw. The mesh LOD program once never received the colour-mode uniforms, so intensity, class and elevation were wrong on 4.06 % of pixels until the parity harness ([Post 11](../post11-image-parity/)) caught it.

## Costs and limits

The per-frame LOD CPU work (selection plus cache bookkeeping) is at most 0.06 ms in the benchmarks. A page-table raster costs one `uvec2` read per workgroup. Block culling ([Post 15](../post15-block-culling-in-the-compute-raster/)) is a feature of loaded clouds: the page raster has no box-cull variant, and the block boxes are built only for clouds that go through the Morton order.

Known compromises, as recorded:

- The cache size is fixed when the cloud opens, and each LOD cloud has its own region.
- **Removing a cloud must drain the staging ring first** (`staging_drain`, Post 06), because the compaction moves the regions that queued copies target.
- A page's unused tail wastes up to 4095 points per node, which is why 20 000-point leaves (five pages) are a better fit than tiny nodes.

## Checklist

1. **Make streamed data look like loaded data**: same pool, same global indices, same shaders.
2. Use **fixed-size pages that equal a whole number of workgroups**; one dispatch plus a `uvec2` table covers any set of pages.
3. Keep the cache **GL-free and testable** with a stand-in pool; audit its invariants over many frames.
4. Evict by **last use with a grace period** that covers asynchronous readbacks.
5. Give every fallback path (multi-draw, mesh) a parity check.

Next: while the octree is still converting, what does the user see? [Post 22](../post22-previewing-while-converting/).

## Where this lives in the code

`src/lod/lod.hpp`, `lod.cpp` (`LodCache`, `lod_frame`, `lod_point`), `src/render/render.cpp` (`raster_pages`, `kRaster` with `PCE_PAGES`, `draw_mesh_pages`, `draw_points_multi`), `src/app/lod/lod_ops.cpp` (the per-frame glue). Tests: `check_lod_cache`, `check_lod_determinism`, `check_lod_budget`, `check_lod_thrash`. Numbers: ROADMAP 6.1.
