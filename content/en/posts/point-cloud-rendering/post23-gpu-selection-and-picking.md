---
title: "Post 23: GPU Selection and Picking"
date: 2026-10-15T17:00:00+03:00
weight: 24
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - selection
  - picking
---

Rendering 100 M points is half of an editor. The other half is *doing something to them*: draw a lasso around a tree, delete it, undo. The acceptance test for selection was "100 M points select in less than one frame". This post covers how a screen-space shape becomes a set of selected points on the GPU, and how a single click finds the point under the cursor, both built on the 64-bit framebuffer of [Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/).

## Selection: rasterize the shape, test the points

The user draws a box or a lasso on the screen. The straightforward way would be to test each of 100 M points against the polygon: expensive per point and awkward on a GPU. The editor flips it: **rasterize the polygon into a bit mask once on the CPU, then each point projects itself and reads one bit.**

```cpp
// Pixels of a w x h target whose centres (x + 0.5, y + 0.5) are inside poly (even-odd rule),
// as bits: pixel p = y*w + x is bit p&31 of word p>>5. Scanline fill.
inline std::vector<uint32_t> polygon_mask(std::span<const glm::dvec2> poly, int w, int h);
```

For each row it collects the polygon edges that cross the row's centre line, sorts the crossings, and fills the pixels between pairs. The even-odd rule means a concave lasso (a star, say) selects correctly. A box is just a 4-vertex polygon. At 2400 x 1350 the fill takes **0.30-0.40 ms**, and the mask is 405 KB: uploaded to an SSBO each time the shape changes. A brute-force `point_in_polygon` lives next to it as the test oracle.

The GPU pass (`kEdit`, `op = select`) is one thread per point:

```glsl
// Same projection, clip test and pixel as the raster.
bool in_mask(uint i, uint g) {
    const uint j = skip + 3 * i;
    const vec4 c = mvp * vec4(pos[j], pos[j + 1], pos[j + 2], 1.0);
    if (!(c.z <= c.w && c.z >= 0.0 && abs(c.x) <= c.w && abs(c.y) <= c.w)) return false;
    const ivec2 px = ivec2((c.xy / c.w + 1.0) * (0.5 * vec2(size)));
    if (px.x >= size.x || px.y >= size.y) return false;
    const uint p = uint(px.y * size.x + px.x);
    return (mask[p >> 5] >> (p & 31u) & 1u) != 0u && ((arg & 4u) == 0u || fb[p].x == g);
}
```

Three design points:

**It reuses the raster's math exactly.** The same `mvp` (camera-relative, per cloud, [Post 04](../post04-camera-relative-rendering-and-reverse-z/)), the same clip test, the same floored pixel. A selected point is therefore a point the user *could have seen* inside the shape, not something computed by a different projection.

**Visible-only selection.** With bit `kVisibleOnly` set, a point counts only if `fb[p].x == g`: it owns its pixel in the compute raster's `fb64` for the current frame. Occluded points behind a wall do not get selected. This is the ID buffer paying for itself: no extra render, one comparison. (A limit recorded in the notes: points seen only through point-size dilation, [Post 12](../post12-point-size-without-sprites/), are not selected, since the visible-only test uses radius 0.) The fb64 read here is declared `uvec2`, so editing needs no 64-bit extension.

**Hidden and deleted points are skipped.** Both are flag bits (Post 24), so the selection tool never grabs what the user cannot see.

Modifiers are an argument: replace, add (Shift), subtract (Ctrl) are `SelectMode` in the op's `arg`; invert, clear, select-all, select class and select-by-mask are the same pass with a different `op`.

### Does the GPU agree with the CPU?

The bench runs each op on the GPU and on a CPU reference with the same float `mvp` and the same mask. Results on 78 M points:

| case | points differing |
|---|---|
| box | 3 of 60.1 M selected |
| concave 12-point star | 0 |
| box subtract | 8 |
| visible-only star | 5 (with fb64 read back) |
| after a cloud removal | 4 |

All sixteen inspected differences sit within 1.2e-4 pixels of a pixel edge (for example `x = 1599.99988`): the GPU's fused multiply-add and the CPU's separate multiply and add land on different sides of the `floor`. That is a rounding difference at the pixel boundary, documented, not a logic bug. Timings (median of 50, op plus delta clear): box 1.22 ms, lasso 1.17 ms, box visible-only 1.11 ms: comfortably inside a frame.

## Picking: the same buffer, one pixel

A click wants one point: its index, absolute coordinates, class and flags. The pick pass is a single-thread compute shader that does what the resolve does at one pixel:

```glsl
uvec2 v = uvec2(0xffffffffu);
for (int r = radius; r <= max(grow, radius) && v.y == 0xffffffffu; ++r)
    for (int y = max(p.y - r, 0); y <= min(p.y + r, size.y - 1); ++y)
        for (int x = max(p.x - r, 0); x <= min(p.x + r, size.x - 1); ++x) {
            const uvec2 f = fb[y * size.x + x];
            if (f.y < v.y || (f.y == v.y && f.x < v.x)) v = f;
        }
stats[4] = (v.y == 0xffffffffu) ? 0xffffffffu : v.x;   // the picked index (or none)
stats[5] = v.y;                                          // its view distance bits
stats[6] = ... class of point v.x;  stats[7] = ... flags of point v.x;
```

It takes the minimum over the **same `(2r+1)²` window the resolve shows**, so the click lands on the point under the cursor *as drawn*, not on a hidden one. An empty window can grow a pixel at a time (`grow`), which panoramas use because the image hides the gaps between points.

**No stall.** The result is four words (index, distance bits, class, flags) written to a small stats buffer, copied into a persistent-mapped buffer, fenced, and read a frame or two later. The UI shows the point when the fence has signalled; nothing waits ([Post 06](../post06-persistent-mapped-staging-ring/)'s pattern). The panel then shows the cloud, the double-precision world coordinate (origin plus pos, [Post 02](../post02-data-layout-for-gpu-point-clouds/)), the colour and intensity from the CPU arrays (never edited), and the class and flags from the GPU (editable).

Verification: a bench picks 125 pixels (100 covered) and compares against a CPU minimum over the read-back `fb64` window: **0 disagreements**; a UI-driven click picks the CPU-expected index.

### A bug the 3x3 window hid

The first version picked **one framebuffer row above the cursor**. The cause: `MousePos` is a window pixel's top-left corner, and the y flip needs its centre. With the default window of 3 px (a 3x3 pick window) the error was invisible on four tiles; on a single tile (`L004`) it picked index 6002867 instead of 6002865. Lesson: a tolerant window can hide an off-by-one. Test pick coordinates at point size 1.

## What if the compute raster is off?

Selection and picking need `fb64`. In `GL_POINTS` mode, the compute raster is run on demand just for them, but where compute is unsupported there is no picking and no visible-only selection ([Post 17](../post17-choosing-a-renderer-at-runtime/) says so at startup). The on-demand path is listed as untested in the notes.

## A caution on timings across machines

The 1.1-1.2 ms selection times are from the Windows 4090 runs. The later "task 2.3" frame-stall table, taken on a loaded Linux machine with `glFinish` around each call, lists a box select at 47-53 ms on 78.6 M points. Those are different drivers, load and measurement methods (wall time with a finish, including CPU work such as the mask fill and upload), so do not compare them directly; they are in the table because that table asks a different question: how long can a frame stall?

## Checklist

1. **Rasterize the shape once, test points against a bitmask**; keep a brute-force oracle.
2. **Use exactly the raster's projection** for everything that must agree with the picture.
3. **Make visibility an ID comparison** against the framebuffer you already have.
4. Read results back **asynchronously** (copy, fence, read later).
5. Expect and document **sub-pixel rounding** differences between GPU and CPU.
6. Test with **small windows** (size 1) so tolerance does not hide coordinate bugs.

Next: what happens to the points once they are selected: [Post 24](../post24-flags-and-class-edits-as-compute-passes/).

## Where this lives in the code

`src/render_model/render_model.hpp` (`polygon_mask`, `point_in_polygon`, `EditOp`, `SelectMode`), `src/render/render.cpp` (`kEdit`, `kPick`, `edit_points`, `pick_point`), `src/app/scene/edit_ops.cpp` (`run_op`). Numbers: ROADMAP 4.2-4.5. Commits 694e40c, f0f65ec.
