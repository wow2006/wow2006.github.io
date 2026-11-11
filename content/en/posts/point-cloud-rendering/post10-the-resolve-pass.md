---
title: "Post 10: The Resolve Pass"
date: 2026-10-14T19:00:00+03:00
weight: 11
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - fragment-shader
---

[Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/) left us with a buffer of 3.24 M 64-bit integers: for each pixel, the nearest point's distance and index. Nothing has a colour yet. The resolve pass turns that buffer into the scene colour and depth, and it does so at **0.036 ms** for the whole frame on the reference machine, against 1.07 ms for the raster.

## Why a fullscreen fragment pass, not another compute shader

A compute shader cannot write a depth attachment (a depth buffer cannot be bound as an image). The resolve needs to produce both colour *and* depth in the scene framebuffer, because things drawn after it with the hardware pipeline, such as the grid, wire boxes, markers and the gizmo, depth-test against it (Post 04). A fragment shader writes `gl_FragDepth`. So the resolve is a draw, with no vertex buffer:

```glsl
// vertex: one triangle that covers the screen
void main() { gl_Position = vec4(vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0, 0.0, 1.0); }
```

```cpp
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);   // make the raster's atomics visible
glUseProgram(program);
glBindVertexArray(pool.vao);                      // core profile needs a VAO even if it has no attributes
glDrawArrays(GL_TRIANGLES, 0, 3);
```

One oversized triangle rather than a quad avoids the diagonal seam and the duplicate fragment work along it. The fixed-function state is the scene state from Post 04: reverse-Z, `GL_GREATER`, depth write on.

## Per pixel: decode the winner

```glsl
uint64_t m = fb[p.y * size.x + p.x];            // (with radius 0; Post 12 widens this)
const uvec2 v = unpackUint2x32(m);              // x = point index, y = view-distance bits
if (v.y == 0xffffffffu) discard;                // nothing landed here: leave the cleared background
const uint i = v.x;
```

An empty pixel still has the all-ones clear value, so `discard` leaves the clear colour and the depth of 0 (infinity in reverse-Z) in place.

## Which cloud does this point belong to?

The pool is shared (Post 03), so the global index must be mapped back to per-cloud state: tint, the z offset for elevation colouring. The CPU builds a small sorted table once per frame:

```cpp
struct CloudRange { uint32_t first, end; float zoff, pad; glm::vec4 tint; };   // 32 B, std430
```

and the shader binary-searches it. `range_index()` in `render_model.hpp` is its CPU twin, tested without GL.

Two housekeeping details matter at scale:

- **The ranges are sorted by `first`**, because scene order is not pool order after clouds are removed or reordered.
- **Adjacent ranges with identical tint and z offset are merged** (`merge_ranges`), so a scene of 40 files with the same settings searches one entry. The buffer starts at 4096 entries and doubles on demand (`bit_ceil`), and the app logs when it grows.

## Fetch only what the mode needs

```glsl
const vec4 c = unpackUnorm4x8(rgbaAt(ci, j));
const uint inten = colorMode == 1 ? (intensity[i >> 1] >> ((i & 1u) * 16u)) & 0xffffu : 0u;
const uint k     = colorMode == 2 ? (cls[i >> 2] >> ((i & 3u) * 8u)) & 0xffu : 0u;
const float z    = colorMode == 3 ? posZ(ci, j) + ranges[lo].zoff : 0.0;
const uint f     = useFlags ? (flags[i >> 2] >> ((i & 3u) * 8u)) & 0xffu : 0u;
fragColor = vec4(shade(point_color(c.rgb, inten, k, z, i), f, ranges[lo].tint), c.a);
```

Each of these is a **random gather**: the pixels' winning indices are unrelated to each other, so each read is a cache miss. The comment in the shader records the measurement: reading all attributes in every mode took RGB fit-all from 1.24 ms to 1.58 ms. Hence the `colorMode == ...` guards. Because SoA arrays are separate buffers (Post 02), a mode that does not need intensity simply does not touch that buffer.

`point_color()` is the single colour function, shared as a GLSL string by this pass and by the `GL_POINTS` vertex shader, so the two renderers cannot drift apart on palettes, ramps or debug modes.

`shade()` applies per-point flag colour (a selected point blends 60 % toward red in every mode) and then the cloud's tint on top. The flag read is guarded by a uniform, `useFlags`, which is off while no flag is set anywhere.

## Chunks again

`rgba` for the whole pool is bound as up to four SSBO windows (Post 03's `resolve_chunks`), since a fragment shader cannot loop over bindings at runtime. The shader picks one by index:

```glsl
uint rgbaAt(uint c, uint j) { return c == 0u ? rgba0[j] : c == 1u ? rgba1[j] : c == 2u ? rgba2[j] : rgba3[j]; }
...
uint ci = 0u, j = i;
if (i >= chunk) { ci = i / chunk; j = i - ci * chunk; }   // one comparison when the pool fits chunk 0
```

Position is bound the same way but only read in elevation mode; in other modes its four bindings point at a 16-byte dummy range so the draw never exceeds a binding limit. The resolve uses 14 SSBO blocks in total (fb64, 4 x rgba, intensity, class, 4 x pos, ranges, flags, and the batch table of the debug mode in Post 08). The driver's fragment-stage limit on the test machine is 16, and the app prints it at startup to confirm that there is room.

## Depth for the rest of the frame

```glsl
gl_FragDepth = ortho ? 1.0 - uintBitsToFloat(v.y) : near / uintBitsToFloat(v.y);   // reverse-Z, same as GL_POINTS
```

The distance stored in the key is converted back into the same reverse-Z depth the hardware would have produced. Then everything works through the normal depth test: the grid sits correctly in front of or behind points, markers occlude properly, and the edge-detecting eye-dome lighting (Post 14) reads a real depth texture. The compute path and the `GL_POINTS` path write the same depth buffer. At fit-all the two depth buffers are bit-identical except for 3 577 of 3.24 M pixels (ROADMAP 3.2: the depth ties, rounding and one-pixel shifts Post 11 examines).

## What the resolve costs

| view | raster | resolve | GPU scene |
|---|---|---|---|
| fit-all | 1.068 ms | **0.036 ms** | 1.241 ms |
| close-up | 1.097 ms | 0.035 ms | 1.269 ms |

The resolve visits each pixel once, no matter how many points there are. That is the invariant that makes the whole design scale: *raster cost grows with points, resolve cost grows with pixels.* The headline comparison, re-verified on a different OS and driver:

| | `GL_POINTS` | compute | ratio |
|---|---|---|---|
| fit-all, 1 px | 23.59 ms | 1.24 ms | 19.0x |
| close-up, 1 px | 2.43 ms | 1.27 ms | 1.9x |
| fly path p99 | 26.2 ms | 1.34 ms | 19.5x |
| fit-all, 3 px | 28.97 ms | 1.25 ms | 23x |

The original "18-20x" claim holds only when most points are on screen. Close-up the advantage is under 2x, before block culling (Post 15) changes that.

## Reuse: the resolve loop is also the picker

The pick pass walks the same window of `fb64` (`min` over the (2r+1)² neighbourhood) to find "the point the resolve shows at this pixel", and writes its index, class and flags to a small buffer. The picking logic and the display logic therefore cannot disagree about which point is under the cursor, because they share the definition.

## Checklist

1. Resolve in a **fullscreen triangle**, so you can write colour *and* depth and keep using the hardware depth test for everything else.
2. Look up per-cloud data by **binary search over a merged, sorted range table**; keep a tested CPU twin.
3. **Fetch the minimum**: one gather per attribute actually in use, behind a mode check. Measure the gathers; they are the resolve's real cost.
4. Convert the key back to the **same depth convention** as the rest of the frame.
5. Put a **memory barrier** between the raster and the resolve. It is the only synchronization the two passes need.

Next: how do you know the new renderer draws the *same picture*? [Post 11](../post11-image-parity/).

## Where this lives in the code

`src/render/render.cpp` (`kResolveVert`, `kResolveFrag`, `resolve_points`, `PCE_COLOR_GLSL`), `src/render_model/render_model.hpp` (`CloudRange`, `range_index`, `merge_ranges`, `resolve_chunks`), `src/app/scene/render_scene.cpp` (building and sorting the ranges). Numbers: ROADMAP 3.2-3.3, BENCH.md F0.1.
