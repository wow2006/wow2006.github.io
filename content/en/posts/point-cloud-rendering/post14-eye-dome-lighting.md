---
title: "Post 14: Eye-Dome Lighting as a Final Composite"
date: 2026-10-14T23:00:00+03:00
weight: 15
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - shading
---

A point cloud has no normals and no surfaces. Unlit, a wall and the tree in front of it are the same flat colour. Computing normals for 100 M points is a project of its own, but there is a cheap trick that needs only the depth buffer: **eye-dome lighting** (EDL), used by Potree and CloudCompare. It darkens a pixel whose neighbours are closer to the camera, which draws a dark halo around depth discontinuities. In this editor it costs about 0.003 ms.

## What it computes

For each pixel, compare its log-depth with eight neighbours on a circle, and sum how much *farther* the pixel is than each neighbour:

```glsl
float logDist(ivec2 p) {
    const float d = texelFetch(depthTex, clamp(p, ivec2(0), textureSize(depthTex, 0) - 1), 0).r;
    return d > 0.0 ? (orthoScale > 0.0 ? (1.0 - d) * orthoScale : -log2(d)) : 1e4;   // background = far
}
void main() {
    const ivec2 p = ivec2(gl_FragCoord.xy);
    const float c = logDist(p);
    float sum = 0.0;
    for (int k = 0; k < 8; ++k) {
        const float a = float(k) * 0.7853982;   // pi / 4
        sum += max(0.0, c - logDist(p + ivec2(round(radius * vec2(cos(a), sin(a))))));
    }
    fragColor = vec4(texelFetch(colorTex, p, 0).rgb * exp(-strength * sum), 1.0);
}
```

`shade = exp(-strength * Σ max(0, log2 d − log2 d_neighbour))`. A pixel at the same depth as all its neighbours has `sum = 0` and keeps its colour. A pixel sitting behind a closer edge gets `sum > 0` and darkens exponentially. The default strength is 40 and the radius 1.4 px (Potree's value), both in the Stats panel.

## Why log depth, and why `near` cancels

With the reverse-Z projection of [Post 04](../post04-camera-relative-rendering-and-reverse-z/), the stored depth is `near / d` for view distance `d`. So

```
-log2(depth) = log2(d) - log2(near)
```

and since `near` appears in every term, it cancels in the differences `c − neighbour`. The shader needs no near-plane uniform and the effect is independent of the camera's near distance. Using the *log* of distance (not the distance) makes the halo scale-free: a step of 10 % in distance darkens by the same amount at 5 m and at 500 m.

For orthographic views depth is linear, `(1 − depth) * orthoScale` gives the distance in a unit that is linearised at the target distance, and the same formula applies.

## Treating the background as far

Pixels with no point have depth 0 (infinity in reverse-Z). The shader maps those to `1e4`, a very large log-distance. The consequence is deliberate and written in the code comment:

- a **drawn pixel next to background** sees a neighbour that is "far": `c - 1e4` is negative, `max(0, ...)` makes it 0, so nothing darkens. Points are never darkened by empty space.
- a **background pixel next to a drawn point** has `c = 1e4`, so its sum is large and it darkens fully. That is the outline: the silhouette of the cloud gets a dark rim against the clear colour, and nothing inside the cloud is darkened by the void.

## A final composite instead of a blit

Normally the scene FBO is blitted to the window. When EDL is on, a fullscreen pass replaces the blit:

```cpp
if (edl && !pano_active) {   // reads the scene target, writes the default framebuffer
    draw_edl(*edl_program, pool.vao, target, edl_strength, edl_radius, ...);
}
```

It reads the scene **colour and depth textures** that the resolve wrote ([Post 10](../post10-the-resolve-pass/)), and writes the default framebuffer. Dear ImGui then draws on top, unshaded. Because the compute resolve writes a real depth buffer, EDL works unchanged for the compute path, the mesh path and `GL_POINTS`. That was the reason to put the effect after the renderer, not inside it.

Costs: nine depth fetches per pixel (the centre plus eight neighbours), a `log2` for each, one colour fetch and one `exp`. Measured on 78.3 M points at size 5:

| renderer | view | blit | EDL |
|---|---|---|---|
| compute | fit | 1.278 ms | 1.282 ms |
| compute | close | 1.254 ms | 1.257 ms |
| `GL_POINTS` | fit | 43.254 ms | 43.257 ms |
| `GL_POINTS` | close | 2.340 ms | 2.345 ms |

The pass replaces a blit and adds 0.003-0.005 ms. The cost depends on the number of *pixels* (3.24 M), never on the number of points, which is the same invariant as the resolve.

## Why it pairs with larger points

EDL is for wide points. A 1-pixel cloud has gaps between points, and the "neighbours" at radius 1.4 are often background, so the effect mostly outlines individual points. The benchmark therefore compares EDL against size-5 RGB, not size 1. With 3 to 7 pixel squares ([Post 12](../post12-point-size-without-sprites/)) the depth buffer is nearly solid and the halo follows real depth steps.

## Checklist

1. **Do it in screen space, after the renderer**, from the depth buffer you already have. It then works for every path.
2. Use **log depth** so the effect is scale-free, and exploit reverse-Z so the near plane cancels.
3. Make the **background infinitely far**: halos around silhouettes, no darkening from the void.
4. Expose **strength and radius**; defaults from a known implementation (Potree) are a safe start.
5. Test it: the harness (Post 11) includes EDL steps compared across renderers.

Next: the post where the camera starts doing less work: block culling, [Post 15](../post15-block-culling-in-the-compute-raster/).

## Where this lives in the code

`src/render/render.cpp` (`kEdlFrag`, `draw_edl`), `src/app/frame.cpp` (the composite choice). Numbers: ROADMAP 3.4.
