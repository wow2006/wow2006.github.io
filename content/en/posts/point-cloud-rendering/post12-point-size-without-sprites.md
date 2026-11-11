---
title: "Post 12: Point Size Without Sprites"
date: 2026-10-14T21:00:00+03:00
weight: 13
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
---

At fit-all, a 1-pixel point cloud looks like dust. Users want to enlarge the points. With `GL_POINTS` that is `gl_PointSize`, and the hardware pays for every covered pixel: the fit-all frame goes from 23.2 ms at 1 px to 68.2 ms at 7 px ([Post 01](../post01-why-gl-points-dies/)). A compute rasterizer writes one pixel per point, so how do you draw a square of 3, 5 or 7 pixels? The answer is to do it in the resolve pass, once per pixel.

## The trick: take the minimum over a window

The raster pass ([Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/)) is unchanged: each point lands in its centre pixel and competes by `atomicMin`. Point size appears only in the resolve ([Post 10](../post10-the-resolve-pass/)), which reads a **window** of `fb64` around each pixel instead of the pixel alone:

```glsl
uint64_t m = 0xffffffffffffffffUL;
for (int y = max(p.y - radius, 0); y <= min(p.y + radius, size.y - 1); ++y)
    for (int x = max(p.x - radius, 0); x <= min(p.x + radius, size.x - 1); ++x)
        m = min(m, fb[y * size.x + x]);
```

Why this is right: a square point of size `2r+1` covers pixel `p` exactly when its centre pixel lies within `r` pixels of `p` in both axes. So the point that GL would show at `p` is the nearest point whose centre pixel is in the window. That is the minimum of the window, because the minimum is the nearest by distance (high bits), with the lower index on ties. One 64-bit `min` per fetched pixel, no per-point work, no sprites.

Properties of this approach:

- **Cost scales with pixels and the window, not with points.** At 78 M points and 3 px the compute scene takes 1.229 ms against 1.249 ms at 1 px, the same within noise.
- **Odd sizes only.** The window is symmetric, so sizes are `2r+1`: 1, 3, 5, 7, 9, 11, 13, 15. The UI slider is over `r` (0 to 7) and shows the pixel size, and the default is 3 px. Even sizes are not offered.
- **Both renderers share the control.** `GL_POINTS` takes `float(2*radius+1)` as its point size, so the slider means the same thing everywhere.
- **Round points are skipped.** A disc would need a distance test inside the window and a matching discard in the `GL_POINTS` fragment shader, so the two paths could keep producing the same image. Squares match by construction.

## What it costs

Measured on 78.3 M points at 2400 x 1350, medians in ms, compute scene (raster + resolve + clear + blit):

| point size | compute fit | compute close | `GL_POINTS` fit | `GL_POINTS` close |
|---|---|---|---|---|
| 1 px | 1.249 | 1.177 | 23.201 | 2.339 |
| 3 px | 1.229 | 1.187 | 28.320 | 2.340 |
| 5 px | 1.278 | 1.254 | 43.254 | 2.340 |
| 7 px | 1.368 | 1.367 | 68.249 | 2.340 |

Between 1 and 7 pixels the compute time grows by 0.12 ms (fit) to 0.19 ms (close). The resolve delta at r=3 is +0.12 ms (fit) and +0.19 ms (close). `GL_POINTS` rises by 45 ms at fit-all and stays flat close-up, because there almost every point is off screen anyway.

The window loop is the naive one, `(2r+1)²` loads per pixel without shared memory, so at 7 px each pixel does 49 reads of 8 bytes. The project kept it: at +0.12-0.19 ms for the largest size there was nothing worth winning yet. If your resolve gets heavier, the standard upgrade is a separable min (a horizontal min into an intermediate buffer, then a vertical min over it, which is exact because `min` over a square is the min of row mins) or a shared-memory tile. Both turn `(2r+1)²` reads into about `2(2r+1)`. The repository implements neither, so treat this as a suggestion.

## Where the approximation shows

Point size is the place where compute and hardware disagree the most, and the project measured it against `GL_POINTS` (pixels differing of 3.24 M, with coverage differences in brackets):

| size | fit | close |
|---|---|---|
| 1 | 3 418 (91) | 2 030 (1 774) |
| 3 | 2 295 (46) | 3 378 (1 823) |
| 5 | 1 845 (32) | 4 857 (1 410) |
| 7 | 1 631 (24) | 7 019 (875) |

Everything here is at the 0.05-0.2 % level, and the coverage differences fall as the point grows. Two differences are structural, not rounding:

1. **Off-screen centres.** The raster discards a point whose *centre* is outside the clip volume. On this NVIDIA driver, a hardware 7-pixel point whose centre is a few pixels off the edge still draws a sliver of its footprint inside the screen (the mesh path only matched `GL_POINTS` after it stopped clipping at the centre). The mesh path widens its clip test by the footprint to match (Post 16); the compute raster does not. The repository's note on this is explicit: wide-point clipping at the screen edge adds no visible band, but that was **inferred from the falling coverage differences, not checked per pixel**.
2. **Depth ties probably get more likely** (my reading, not measured): large near points overlap more, so more pixels have two candidates at nearly the same distance.

## Culling must know the footprint

Anything that decides whether a point can matter has to be aware of the size, or it will cull points whose footprint reaches the screen. In the mesh path the clip test is `|x| <= w * reach` with `reach = 1.0 + pointSize / viewport`, and the task shader divides the matrix rows by `reach` for the block test (Post 15 and 16). For the compute path the equivalent is: a centre outside the screen is dropped, and that is the known difference above.

## Picking uses the same window

When the user clicks, the pick pass takes the min of `fb64` over the same `(2r+1)²` window the resolve shows, and reports that point. The click lands on the point under the cursor *as drawn*, not on a point that is hidden behind a bigger one.

## Checklist

1. If your rasterizer writes one pixel per point, **do point size in the resolve** with a window min. Keep the raster size-independent.
2. **Squares first.** Make the compute and hardware paths match exactly, then consider discs.
3. Know the cost per size, and decide whether the naive loop is good enough before optimizing it.
4. **Footprint-aware culling** everywhere a point can be dropped.
5. Reuse the display window for picking.

Next: switching colour modes without re-uploading anything: [Post 13](../post13-colour-modes-without-re-upload/).

## Where this lives in the code

`src/render/render.cpp` (`kResolveFrag` window loop, `draw_points` point size, the pick pass), `src/app/scene/render_scene.cpp` (`point_radius`). Numbers: ROADMAP 3.3b.
