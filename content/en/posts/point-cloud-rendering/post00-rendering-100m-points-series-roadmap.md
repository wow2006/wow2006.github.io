---
title: "Post 00: Rendering 100M+ Points with Modern OpenGL: Series Roadmap"
date: 2026-10-14T09:00:00+03:00
weight: 1
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
---

This series covers rendering very large point clouds, 100M+ points, with OpenGL 4.6. Every post is drawn from the development of a real point cloud editor, so the numbers are measurements, not estimates. The reference machine is an RTX 4090 rendering 78.3M points at 2400x1350, where plain `GL_POINTS` takes 23.6 ms per frame and our compute rasterizer takes 1.24 ms.

## The Roadmap

### Part 1: Foundations

1. **Why GL_POINTS dies at 78 M points.** The frame budget, and the rules for measuring honestly: vsync off, fence pacing, logged GPU clocks, and GL_TIME_ELAPSED medians.
2. **Data layout for GPU point clouds.** Structure of arrays, a double origin with float offsets, and separate pos, rgba, intensity and cls buffers.
3. **One big pool, not one buffer per cloud.** Per-attribute SSBO pools, 178M-point chunks bound with glBindBufferRange, offset alignment, and std430 `float pos[]` instead of `vec3[]`.
4. **Camera-relative rendering and reverse-Z.** Double-precision model matrices cast to float, glClipControl ZERO_TO_ONE, GL_GREATER, and an infinite far plane.
5. **Morton order plus a 128-point block shuffle.** Spatially coherent blocks from a parallel radix sort, taking GL_POINTS from 58 ms to 24 ms.

### Part 2: Getting Data to the GPU

6. **Persistent-mapped staging ring.** glBufferStorage, fences, packing small uploads, and the NVIDIA driver traps.
7. **Loading 100M points without freezing.** Streaming batch readers and parallel PLY, LAS and LAZ decoding.
8. **Pool order and upload batches.** Debugging with the Order-block and Upload-batch colour modes.

### Part 3: The Core Idea, a Compute Rasterizer

9. **Software-rasterizing points with 64-bit atomics.** atomicMin on `depth << 32 | point index`, with GL_ARB_gpu_shader_int64 and GL_NV_shader_atomic_int64, and the same buffer doubling as an ID buffer for picking.
10. **The resolve pass.** A fullscreen triangle, gl_FragDepth, and a tint range table: 1.24 ms against 23.1 ms, 18.6x faster.
11. **Image parity.** Proving the rasterizer matches GL, with depth ties, window-coordinate snapping and pixel diffs.
12. **Point size without sprites.** A minimum over a (2r+1)^2 window: +0.12 ms at 7 px, where GL_POINTS goes from 23 ms to 68 ms.
13. **Colour modes with no re-upload.** One shared GLSL function for RGB, intensity, class and elevation.
14. **Eye-dome lighting as a final composite.** Log-depth, 8 neighbours, about 0.003 ms.
15. **Block culling in the compute raster.** The Morton blocks as a free culling hierarchy.

### Part 4: The Alternative and the Fallback

16. **Mesh shaders for points.** GL_NV_mesh_shader with task-shader block culling and per-pixel dedup in the warp: 33-35x faster close up, but still 1.4x slower than compute when fitting the whole cloud.
17. **Choosing a renderer at runtime.** Compute, mesh as opt-in, and a GL_POINTS fallback, with extension checks and the limits past 715M points.

### Part 5: Scaling Past VRAM

18. **An out-of-core octree for point clouds.** The .pco format, screen-space-error node selection, a point budget, and a page cache.
19. **Quantized nodes.** 12 bytes per point on the source's integer grid.
20. **Streaming LOD from a cold disk, 12x faster.** Prefetch, parallel fill, and cold versus warm measurement.
21. **Feeding the same shaders from a streamed page table.**
22. **Previewing while converting.**

### Part 6: Editing on the GPU

23. **GPU selection and picking.** Box and lasso selection, and ID-buffer picking with copy and fence instead of a synchronous readback.
24. **Flags and class edits as compute passes.** The GPU is authoritative, and CPU arrays stay stale until a snapshot.
25. **Undo and redo as GPU XOR deltas.** VRAM-capped history, remapped through compaction.
26. **Edits on streamed clouds.** Op log replay as nodes load, and the density holes that deletes leave behind.
27. **kNN outlier removal on a GPU grid.** Run in time-sized slices to avoid GL_GUILTY_CONTEXT_RESET and respect the 50 ms no-freeze rule.

### Part 7: Engineering Discipline

28. **Benchmarking a renderer honestly.** Bench modes, a CPU mirror to verify GPU edits, and a benchmark row for every claim.
29. **Profiling with Tracy and Nsight.**
30. **Putting it together: a 235M-point frame budget.**

## Core Path

If you only have time for a few posts, read posts 1, 3, 5, 6, 9, 10, 16 and 18. They cover the essentials: the budget, the pool, the coherent ordering, streaming uploads, the compute rasterizer and its resolve pass, the renderer choice, and the out-of-core octree.
