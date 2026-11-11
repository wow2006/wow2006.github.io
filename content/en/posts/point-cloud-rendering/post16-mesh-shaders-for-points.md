---
title: "Post 16: Mesh Shaders for Points (GL_NV_mesh_shader)"
date: 2026-10-15T10:00:00+03:00
weight: 17
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - mesh-shader
---

The compute rasterizer in Posts 09 and 10 skips the hardware pipeline entirely. That raises an obvious question: could the *hardware* pipeline be fed better instead? Mesh shaders let a shader produce primitives directly, with no vertex fetch stage and no index buffer, and a task shader in front can cull whole groups. The project built that path with `GL_NV_mesh_shader`, measured it, and kept it **optional**. This post covers what was built, how it improved by 33x, and why compute is still the default.

All timings below are labelled with their machine. The compute-versus-mesh results come from two GPUs: an RTX 4090 (first, naive version) and an RTX 5070 Ti Laptop GPU (the reworked version). A laptop GPU throttles, so the notes say absolute times drift about 20 % between runs and every comparison was interleaved (old, new, compute, back to back, twice).

## The shape of the pipeline

```
task shader (1 workgroup = 32 lanes = 32 blocks)   -> culls blocks by their box, compacts survivors
mesh shader (1 workgroup = 32 lanes = 128 points)  -> per point: clip, dedupe by pixel, emit survivors as point primitives
fragment shader                                    -> fetch attributes by index, colour (same function as everywhere)
```

The mesh shader emits `points` primitives with up to 128 vertices per workgroup. The output goes through the normal rasterizer, the normal early depth test and `GL_GREATER`, so the result lands in the scene FBO like `GL_POINTS` would, with no resolve pass.

## Version 1: naive, and 20x slower

The first mesh path simply emitted every point as a primitive. On the RTX 4090, 78.3 M points, 1 px (BENCH F0.3, two 600-frame runs):

| view | compute | mesh |
|---|---|---|
| fit | 1.22 ms | 24.25 ms (19.9x slower) |
| close | 1.25 ms | 1.76 ms (1.41x slower) |

The mesh route needed to process and rasterize the whole point set, like `GL_POINTS`, and the commit made it opt-in, not default (d473440). That would have been the end of the story in many projects. Instead the numbers pointed at where the time went: per-fragment work and primitives that lose the depth test.

## Version 2: four changes

**1. One warp per 128 points, loads first.** A workgroup of 32 lanes handles a 128-point block in 4 rounds. All four rounds' position and flag loads are issued before any subgroup operation, so each lane has 4 memory requests in flight instead of one:

```glsl
vec3 pts[4]; uint fl[4];
for (uint q = 0u; q < 4u; ++q) {            // all reads first, none behind a subgroup op
    const uint i = first + 32u * q;
    ...
    pts[q] = vec3(pos[pj], pos[pj + 1u], pos[pj + 2u]);
}
```

**2. Drop what cannot draw.** Hidden, deleted and off-screen points are removed in the shader. The clip test uses the point's footprint (`reach = 1.0 + pointSize / viewport`), since the hardware draws a wide point whose centre is just off screen.

**3. Per-pixel dedup inside the warp.** This is the idea that did most of the work. At fit-all there are about 24 points per pixel; most of them are going to lose the depth test. Inside a round, the 32 lanes are grouped by pixel with `subgroupPartitionNV`, and a partitioned max over the depth bits picks the nearest per group. Only lanes at that max are emitted:

```glsl
const uvec4 part = subgroupPartitionNV(!valid ? 0xffffffffu : edge ? 0x80000000u | l : uint(px.y << 16 | (px.x & 0xffff)));
const uint d = valid ? floatBitsToUint(c.z / c.w) & ~31u : 0u;        // window depth >= 0 orders like its bits
const bool keep = valid && d == subgroupPartitionedMaxNV(d, part);
const uvec4 ballot = subgroupBallot(keep);
if (keep) { const uint o = n + subgroupBallotExclusiveBitCount(ballot); /* emit vertex o */ }
n += subgroupBallotBitCount(ballot);
```

Details that came from measuring and from the image diff:

- The low 5 bits of the depth are cleared, so near-ties all survive and the hardware depth test settles them exactly. An earlier variant with two reductions (max depth, then min lane) was 6.04 ms against 4.20 ms with the single truncated-depth reduction; no dedup at all was 4.55 ms.
- A centre within 1/128 pixel of a pixel edge gets a partition of its own, because the hardware snaps centres to a sub-pixel grid, and such a point may land in the neighbour. The first version of the dedup lacked this guard, clipped x and y at the centre, and carried a packed colour (below); it differed from `GL_POINTS` in 0.01-0.6 % of pixels, and each of the three fixes was isolated with a variant that had only it removed.
- Survivors on one 10.3 M-point tile: 7.7 % of the points at fit-all, 1.6 % close-up. Everything else never reaches the rasterizer.

**4. Emit an index, not a colour.** Each survivor carries only its 4-byte point index. The fragment shader fetches the attribute and calls `point_color()`, so colour work runs only for fragments that pass the early depth test:

| vertex payload | fit, 1 px (p50) | fit, 7 px | close |
|---|---|---|---|
| `vec4` colour (16 B per vertex, exact) | 7.631 ms | 12.652 | 0.267 |
| `packUnorm4x8` (4 B, **not exact**) | 5.877 ms | | |
| **index only (kept)** | **3.763 ms** | **11.700** | **0.232** |

The packed colour was faster than the `vec4` but rounds differently from the RGBA8 target, one of the three causes of the first version's differing pixels, so it was rejected on correctness, not speed.

## Task shader: block culling

The same boxes from [Post 15](../post15-block-culling-in-the-compute-raster/), in a different place. One task workgroup covers 32 blocks; each lane tests one block's box against the clip planes (pushed out by the footprint), and the survivors are compacted with a ballot into the payload. Only surviving blocks launch a mesh workgroup:

```glsl
taskNV out Task { uint meshlet[32]; } task;
...
const uvec4 ballot = subgroupBallot(keep);
if (keep) task.meshlet[subgroupBallotExclusiveBitCount(ballot)] = m;
if (gl_LocalInvocationID.x == 0u) gl_TaskCountNV = subgroupBallotBitCount(ballot);
```

An outside block launches no mesh workgroup and reads no point. Unlike the compute raster ([Post 15](../post15-block-culling-in-the-compute-raster/)), the mesh path skips the task shader when the whole cloud's box is inside the view (`box_inside`), because it costs about 0.65 ms at fit-all when it culls nothing. With culling the close-up went from 2.39 to 0.49 ms in the intermediate measurement.

## Results (RTX 5070 Ti Laptop, 78.3 M points, 600 frames, GPU scene p50: round 1 / round 2, interleaved)

| path | old mesh | new mesh | compute | mesh speed-up | new mesh vs compute |
|---|---|---|---|---|---|
| fit | 29.7 / 34.8 ms | **5.7 / 5.9 ms** | 3.9 / 4.3 ms | 5.2x / 5.9x | 1.48x / 1.39x **slower** |
| close | 8.46 / 8.54 ms | **0.24 / 0.26 ms** | 2.18 / 2.87 ms | 35x / 33x | 9.0x / 11x **faster** |

So after the rework mesh shaders **beat compute close-up by about an order of magnitude** and still lose by about 1.4x at fit-all. (The compute numbers on this machine are slower than on the 4090 because it is a throttled laptop part; compare ratios, not absolute times, across machines. I did not find a re-run of the reworked mesh path on the 4090 in the repository.)

## Why the default stays compute

1. **Fit-all is still slower.** When every point is visible the mesh path has to run the hardware rasterizer for each surviving primitive. Compute resolves 24 points per pixel with atomics and shades each pixel once.
2. **No per-pixel point index.** The mesh route uses ordinary depth testing and gives no `fb64` to read back. Picking, visible-only selection and the edit passes all need that buffer (Post 23). So the editor keeps the compute path for editing and on-demand picking and uses mesh only for drawing.
3. **Tie-breaking differs.** The deterministic lower-index winner is a property of the atomic key; the hardware depth test keeps the first drawn. The mesh path matches `GL_POINTS`, not compute, on ties.
4. **Portability.** `GL_NV_mesh_shader` is an NVIDIA extension.

It also draws the same image as `GL_POINTS`: in 20 steps (fit, close, ortho; sizes 1 to 7; four colour modes; EDL; grid) **0 of 1 440 000 pixels differ, largest channel difference 0** (Post 11). One residual on a single tile: 2 pixels off by 1 LSB in close-up elevation, because `GL_POINTS` interpolates its colour varying; the cause was confirmed by a variant that interpolates the same way.

Along the way the image check found a pre-existing bug: the LOD mesh program never received the colour-mode uniforms, so intensity, class and elevation were wrong on 4.06 % of pixels. Fixed with the same harness.

## Checklist

1. **Measure the naive version honestly**, and keep the numbers, even when it loses 20x.
2. Look for work that reaches the rasterizer and **loses the depth test**: dedupe it inside the warp with subgroup partitions.
3. Emit **small payloads**. An index plus a fragment-time fetch beat colour-in-vertex.
4. Put the culling in a **task shader**, but only where it culls something.
5. Make the **image check** a gate for every shader tweak, and isolate each fix with a variant.
6. Keep the path **optional** if it cannot do everything the default can.

Next: how the editor picks among these renderers at runtime, and tells you why: [Post 17](../post17-choosing-a-renderer-at-runtime/).

## Where this lives in the code

`src/render/render.cpp` (`kMesh`, `kMeshTask`, `kMeshFrag`, `create_mesh_program`, `draw_mesh_points`, `draw_mesh_pages`). Numbers: BENCH.md "F0.3 - compute vs mesh shader" and "Mesh shader pipeline", ROADMAP 3.2. Commits bc63de5, d473440, 6e2911d.
