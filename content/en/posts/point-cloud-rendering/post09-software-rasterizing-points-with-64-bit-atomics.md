---
title: "Post 09: Software-Rasterizing Points with 64-bit Atomics"
date: 2026-10-14T18:00:00+03:00
weight: 10
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - compute-shader
  - atomics
---

[Post 01](../post01-why-gl-points-dies/) ended with a diagnosis: at fit-all there are about 24 points per pixel, and the hardware rasterizer pays full price for the 23 that lose. This is the post where we stop asking it to. The compute rasterizer replaces "draw a point, depth-test it" with a single question per point: *is this the nearest point so far at its pixel?* One 64-bit `atomicMin` answers it.

## The idea in one line

Keep, for every pixel, one 64-bit integer: the distance of the nearest point seen so far in the high 32 bits, and that point's global index in the low 32 bits. Every point does `atomicMin` on its pixel's integer. When all points are done, each pixel holds the winner.

```
fb64[pixel] = (floatBits(view_distance) << 32) | global_point_index
```

Why this works:

- **Positive IEEE floats order like their bit patterns.** For non-negative floats, comparing the `uint32` bit pattern gives the same order as comparing the float. So a smaller distance gives a smaller high word, and `min` on the 64-bit value selects the nearest point.
- **The low word is a free tie-break.** When two points have the same distance, the one with the lower index wins. That makes the image deterministic regardless of thread scheduling, which is something `GL_POINTS` does not guarantee (it keeps whichever was drawn first).
- **The winner's identity is in the buffer.** Not a colour: the index. Colour is fetched once per *pixel* afterwards (Post 10), not once per *point*. With 24 points per pixel, that is a 24x reduction in attribute reads. And it doubles as an ID buffer for picking (Post 23).

## What you need

Two extensions: `GL_ARB_gpu_shader_int64` (64-bit integer types in GLSL) and `GL_NV_shader_atomic_int64` (`atomicMin` on `uint64_t` in a storage buffer). The app checks both at startup, prints them, and falls back to `GL_POINTS` if either is missing:

```
GL_ARB_gpu_shader_int64:   yes
GL_NV_shader_atomic_int64: yes
```

Note this is an NVIDIA-extension design. The point of this series is to show the technique; on hardware without 64-bit atomics you would split into a depth pass and an ID pass, with a second dispatch and more traffic, and the post does not cover that path.

The framebuffer is a plain SSBO of `width * height` 64-bit values:

```cpp
glNamedBufferStorage(t.fb64, size_t(w) * h * sizeof(uint64_t), nullptr, BufferStorageMask{});
// every frame, before the raster:
const uint32_t ones = ~0u;
glClearNamedBufferData(target.fb64, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &ones);   // all bits set = "empty"
```

At 2400 x 1350 that is 3.24 M pixels x 8 B = 26 MB, cleared to all-ones (the largest possible key, so any real point beats it). The clear is counted inside the raster time in the benchmark tables.

## The kernel

One thread per point, 128 threads per workgroup (this number returns in Post 15). Condensed from the project's shader:

```glsl
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_NV_shader_atomic_int64 : require
layout(local_size_x = 128) in;
layout(std430, binding = 0) restrict readonly buffer Pos { float pos[]; };   // float, not vec3 (Post 03)
layout(std430, binding = 1) restrict buffer Fb { uint64_t fb[]; };
layout(location = 0) uniform mat4 mvp;
layout(location = 1) uniform uint count;   // points in this dispatch
layout(location = 2) uniform uint skip;    // floats before the first point in Pos
layout(location = 3) uniform uint base;    // global index of the first point
layout(location = 4) uniform ivec2 size;
layout(location = 5) uniform float orthoK; // 0 perspective, 1 orthographic

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    // (compiled variant PCE_FLAGS: skip hidden / deleted points here)
    const uint j = skip + 3 * i;
    const vec4 c = mvp * vec4(pos[j], pos[j + 1], pos[j + 2], 1.0);
    // Same clip test as GL_POINTS; also rejects w <= 0 and NaN.
    if (!(c.z <= c.w && c.z >= 0.0 && abs(c.x) <= c.w && abs(c.y) <= c.w)) return;
    const ivec2 px = ivec2((c.xy / c.w + 1.0) * (0.5 * vec2(size)));   // viewport transform, floored
    if (px.x >= size.x || px.y >= size.y) return;
    const uint64_t v = packUint2x32(uvec2(base + i, floatBitsToUint(c.w - orthoK * c.z)));
    const uint p = uint(px.y * size.x + px.x);
    if (v < fb[p])             // a stale read is >= the current value, so skipping is always safe
        atomicMin(fb[p], v);
}
```

Walking through the parts that are not obvious:

**The key is the view distance `w`, not the depth.** With the projection from [Post 04](../post04-camera-relative-rendering-and-reverse-z/), perspective `w_clip` equals the distance along the view axis, which is positive for visible points. That is a float whose bits sort correctly. For orthographic projection `w` is 1, so the key is `1 - z` instead (`orthoK = 1`), which also grows with distance. `packUint2x32(uvec2(index, distanceBits))` puts the index in the low word.

**The clip test is the one `GL_POINTS` would apply** (`z <= w`, `z >= 0`, `|x|,|y| <= w`). It also drops points behind the camera and NaNs, because every comparison with NaN is false and the whole expression is negated.

**Read before atomic.** `fb[p]` is read with a plain load first, and the atomic is issued only if this point could win. This is safe because the value in `fb[p]` only ever decreases during the pass, so a stale read is greater than or equal to the current value, and if we would lose against the stale value we also lose against the current one. At 24 points per pixel this removes most atomics from the common case, where most points lose.

**One point per thread.** The project tried 2, 4, 8, 16 and 32 consecutive-strided points per thread. Results (fit/close medians, ms): 1: 1.238/1.174, 2: 1.236/1.176, 4: 1.232/1.177, 8: 1.231/1.180, 16: 1.237/1.186, 32: 1.264/1.210. Two to sixteen are within 0.6 %, 32 is slower, so the simplest kernel stayed.

**Compile-time variants, not uniforms.** The flag test (`flags[...] & (hidden|deleted)`) exists only in the `PCE_FLAGS` variant of the shader. A uniform-gated version cost 0.08 ms at 78 M points even when no point was hidden. Rare features that sit in the hot loop get their own program; the app keeps the flag-free variant until some bit is set anywhere.

## The host side

```cpp
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);     // before the resolve reads fb64 (Post 10)
```

For each visible cloud the raster does the loop from Post 03: bind a window of `pos` with `glBindBufferRange`, set `count`, `skip` and `base`, `glDispatchCompute((count + 127) / 128, 1, 1)`, advance. The `mvp` is the camera-relative matrix from Post 04, built per cloud in double and cast to float. Clouds are processed one after another, and nothing in the kernel is cloud-specific except the matrix, so several clouds of different origins end up in the same `fb64`.

The kernel never writes colour, depth or any attribute. That is the architectural point: **the raster pass only decides who wins.**

## What it costs

Toronto, 78.3 M points, 2400 x 1350, RTX 4090, median of 600 frames, 1 px points:

| view | raster (incl. 26 MB clear) | total GPU scene | `GL_POINTS` scene |
|---|---|---|---|
| fit-all | 1.068 ms | 1.241 ms | 23.592 ms |
| close-up | 1.097 ms | 1.269 ms | 2.430 ms |

The raster is about 86 % of the compute path's scene time. Two observations from the arithmetic, which I am labelling as inference, since the repository has no profiler counters for them:

1. **The raster is near the memory bandwidth limit.** 78.3 M points x 12 B = 940 MB of positions read in 1.07 ms is about 880 GB/s. The RTX 4090's public peak memory bandwidth is about 1 TB/s. Per frame, the kernel reads every position exactly once, mostly sequentially (Morton blocks, Post 05), and almost never writes. `GL_POINTS`, by comparison, moved the same data at roughly 54 GB/s (Post 01).
2. **Close-up costs the same as fit-all.** The kernel still reads and transforms every point, even those far outside the frustum: 1.097 vs 1.068 ms. That is exactly why block culling (Post 15) later cuts the close-up raster from 1.10 to 0.344 ms: it skips the *read*, not just the atomic.

## Limits worth knowing

- **Indices are 32-bit.** The low word holds the global pool index, so a pool is limited to about 4.29 billion points. The project's tested scale is 235 M.
- **No blending, no anti-aliasing, no per-point colour interpolation.** A point is a pixel (or a square, Post 12). It is a point cloud viewer, not a splat renderer.
- **Float distance precision.** 32-bit float keys have 24 significant bits, so two points whose distances differ by less than about one part in 8 to 16 million round to the same key and tie, and the lower index wins.
- **Hardware coverage differences.** `GL_POINTS` snaps window coordinates to sub-pixel fixed point; the shader floors a float. That makes 1-pixel shifts at silhouettes. Post 11 measures it.

## Summary

1. A pixel's best point is `min(distance_bits << 32 | index)`. Order-preserving float bits plus an index tie-break make the answer deterministic.
2. One thread per point, read-before-atomic, a camera-relative `mvp` per cloud, windows of the pool bound with `glBindBufferRange`.
3. The kernel resolves visibility and defers *everything else* to the next pass, which touches each pixel once instead of each point.
4. On the reference machine that takes 78 M points from 23.6 ms to a 1.07 ms raster, and the kernel is close to bandwidth-bound.

Next: turning those 3.24 M 64-bit values into a picture: [Post 10](../post10-the-resolve-pass/).

## Where this lives in the code

`src/render/render.cpp` (`kRaster`, `raster_points`, `scene_target_resize`), `src/app/scene/render_scene.cpp` (`raster_all`, the clear), `src/app/gl/gl_info.cpp` (extension checks). Numbers: ROADMAP 3.2, BENCH.md F0.1.
