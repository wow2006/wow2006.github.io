---
title: "Post 13: Colour Modes Without Re-Upload"
date: 2026-10-14T22:00:00+03:00
weight: 14
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - shaders
---

A point cloud usually carries more than a colour: laser intensity, an ASPRS or dataset classification, and a height that makes terrain readable. A viewer should flip between those views instantly, which on a 100 M-point cloud rules out re-uploading anything. This post shows how the editor switches between RGB, intensity, class and elevation (and two debug modes) with one uniform.

## Everything is already on the GPU

The pool from [Post 03](../post03-one-big-pool/) has one buffer per attribute: `pos`, `rgba`, `intensity` (u16), `cls` (u8) and `flags`. They are all uploaded at load, so a colour mode is a **shader decision**, not data movement. "Done when: switchable from UI, no re-upload" was the acceptance test.

The cost of that choice is VRAM: 2 B/point for intensity and 1 B/point for class, always resident. For 78 M points that is 157 MB + 78 MB, a small price against the 940 MB of positions.

## One colour function for both renderers

```glsl
layout(location = 10) uniform int colorMode;   // 0 rgb, 1 intensity, 2 class, 3 elevation, 4 batch, 5 block
layout(location = 11) uniform vec2 colorRange; // intensity or scene-space z [lo, hi]
layout(location = 100) uniform uint classColor[256];

vec3 point_color(vec3 rgb, uint intensity, uint cls, float z, uint idx) {
    if (colorMode == 4) return batch_color(idx);      // debug (Post 08)
    if (colorMode == 5) return block_color(idx);
    if (colorMode == 1)
        return vec3(clamp((float(intensity) - colorRange.x) / max(colorRange.y - colorRange.x, 1.0), 0.0, 1.0));
    if (colorMode == 2) return unpackUnorm4x8(classColor[cls & 255u]).rgb;
    if (colorMode == 3) {   // blue > green > yellow > red
        const float t = 3.0 * clamp((z - colorRange.x) / max(colorRange.y - colorRange.x, 1e-6), 0.0, 1.0);
        return t < 1.0 ? vec3(0.0, t, 1.0 - t) : t < 2.0 ? vec3(t - 1.0, 1.0, 0.0) : vec3(1.0, 3.0 - t, 0.0);
    }
    return rgb;
}
```

This text lives in one C++ string, `PCE_COLOR_GLSL`, concatenated into the `GL_POINTS` vertex shader, the compute resolve and the mesh fragment shader. Three consumers, one definition, so [Post 11](../post11-image-parity/)'s parity check can compare renderers on every mode with no second implementation to drift. The tint (a per-cloud colour mix) goes on top in every mode, and selected points blend 60 % toward red on top of that.

## The four modes and their normalisation

**RGB.** The packed RGBA8 from the file. LAS 16-bit colours are shifted down by 8 bits only if any channel exceeds 255.

**Intensity.** Raw intensity has outliers, and a min/max mapping made Toronto's median intensity 4 % grey: values run 1 to 255 with a median of 10. So the range is the **1st to 99th percentile**, measured on every 61st point and merged across clouds at ingest (Post 07). That keeps contrast without sorting 78 M values.

**Classification.** A 256-entry palette uniform array. Toronto's labels are mapped through `Colors.xml` at `label / 8` for labels 0 to 8, with anything above shown red. The same palette is used for LAS classes (the roadmap notes that ASPRS colours are not mapped yet). The class panel added later can hide classes, which the raster and mesh shaders handle with a class mask.

**Elevation.** Height ramp blue, green, yellow, red, over the scene-space z range of the loaded bounding boxes. Per-point z is `pos.z + cloud z offset` (`zoff`), a per-cloud float from the range table of Post 10. The code notes the shortcut: only the transform's translation counts, which is exact for rotations about the vertical axis; a tilted cloud would need z from the full transform.

## The costs are in the gathers

The resolve fetches one random value per pixel for each attribute it needs. The first implementation fetched all of them every time. The rule that came out of measuring:

```glsl
const uint inten = colorMode == 1 ? (intensity[i >> 1] >> ((i & 1u) * 16u)) & 0xffffu : 0u;
const uint k     = colorMode == 2 ? (cls[i >> 2] >> ((i & 3u) * 8u)) & 0xffu : 0u;
const float z    = colorMode == 3 ? posZ(ci, j) + ranges[lo].zoff : 0.0;
```

Reading all three in RGB mode took fit-all from 1.24 ms to 1.58 ms. With the guards:

| mode (size 1) | compute fit | compute close |
|---|---|---|
| rgb | 1.249 ms | 1.177 ms |
| intensity | 1.401 ms | 1.199 ms |
| class | 1.330 ms | 1.191 ms |
| elevation | **1.789 ms** | 1.232 ms |

Elevation is the outlier: **+0.54 ms** at fit-all, because it reads a random 4-byte float from the 940 MB position buffer for every pixel. The notes list two fixes if it ever matters: reconstruct z from the pixel position and the depth (no gather, but wrong by up to half a pixel's footprint), or keep a z-only copy next to `rgba`. The second would cost 4 B/point more VRAM. This is a typical engineering position: the cost is measured, the options are written down, and nothing was built because 1.79 ms is fine.

Elevation also has a structural cost. It reads `pos`, so the resolve must bind all of position as chunks (Post 03), and the compute path then reaches 715 M points in every mode instead of only the RGB one.

## Small attributes packed into words

Both buffers are read as `uint[]` words (Post 03): intensity two per word, class four per word, using shifts. The colour in `rgba` is also one `uint`. With `std430` and 4-byte-rounded buffers there are no 8- or 16-bit storage extensions to depend on.

## Parity across modes

The comparison against `GL_POINTS` at size 1 gave:

| mode | fit differing pixels | close differing pixels |
|---|---|---|
| rgb | 3 418 | 2 030 |
| intensity | 2 973 | 2 064 |
| class | 183 | 1 837 |
| elevation | 1 228 | 1 942 |

All are the depth-tie and 1-pixel-shift pixels from Post 11. The mesh shader reaches 0 differing pixels in every mode (Post 16).

## Checklist

1. **Keep every attribute resident** and choose the colour in the shader, so mode changes are free.
2. **One shared GLSL function** for every renderer, via string concatenation.
3. **Normalise robustly**: percentiles on a stride sample, not min/max.
4. **Gather only what the mode needs**, and measure what each mode costs.
5. Write down the next optimisation, with its price, and do not build it until the number demands it.

Next: shading that makes the colour readable without lights: [Post 14](../post14-eye-dome-lighting/).

## Where this lives in the code

`src/render/render.cpp` (`PCE_COLOR_GLSL`, `point_color`), `src/app/loading/ingest.cpp` (intensity percentile), `src/core/classes.hpp`, `src/app/scene/render_scene.cpp` (colour range). Numbers: ROADMAP 3.3 and BENCH.md.
