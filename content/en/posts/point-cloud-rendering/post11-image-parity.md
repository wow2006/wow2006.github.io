---
title: "Post 11: Image Parity, Proving Your Rasterizer Draws the Same Picture"
date: 2026-10-14T20:00:00+03:00
weight: 12
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - testing
---

A renderer that is 19x faster is worthless if it draws something different. Posts 09 and 10 replaced the hardware pipeline with an atomic and a resolve. This post is about the harness that checks the replacement: keep the old renderer around as a reference, diff the two images, classify every differing pixel, and decide which differences you accept.

## Keep the slow renderer as the oracle

`GL_POINTS` is not deleted. It stays as a fallback (Post 17) and as the **reference image** for every other path. The editor's `--bench` mode renders the same camera view with each renderer, reads the pixels back, and compares:

```cpp
const auto [diff, bg] = image_diff(bench_img, bench_ref);   // pixels whose RGB differ, and how many of those are background
std::println("bench {} diff vs GL_POINTS: {} of {} pixels differ ({:.3f}%), {} of them coverage", ...);
```

`image_diff` returns two numbers: how many pixels differ, and how many of those are **coverage** differences, where one image has a point and the other has the background. That split is the key idea. A pixel that has a point in both images but a slightly different colour is a depth tie or a rounding difference. A pixel with a point in one and background in the other means a point is **missing or extra**, which is a real bug candidate.

The harness compares at point size 1 first (so a point is exactly one pixel), then at sizes 3, 5 and 7, in four colour modes, with and without EDL and the grid, in perspective and orthographic views. The first run on the 78 M-point scene, at 2400 x 1350, gave:

| view | pixels differing (of 3.24 M) | of those, coverage |
|---|---|---|
| fit-all, size 1 | 3 418 (0.105 %) | 91 |
| close-up, size 1 | 2 030 (0.063 %) | 1 774 |

## A bug in the checker, not the renderer

The first version of this table said that none of the differing pixels were coverage differences. That was wrong, and the reason is a lesson in comparing images. The harness cleared to a background colour of `0.1`, then looked for background pixels as RGBA value `26`. But `0.1 * 255 = 25.5` rounds to **25** when it is stored in an RGBA8 target. The check never found a single background pixel, so "coverage" was always zero. The fix was one commit: clear to an exact 8-bit colour (`26/255`), and write a comment saying why.

```cpp
// Exact in RGBA8, so --bench can spot background: 0.1f read back as 25, not 26, so coverage was never counted.
const float bg[4] = {26 / 255.0f, 26 / 255.0f, 31 / 255.0f, 1.0f};
```

After the fix, the numbers above were re-measured. If you build a diff tool, give it a way to fail loudly: a test that deliberately breaks the renderer and confirms the diff notices (below).

## What the remaining differences are

Roughly 0.1 % of pixels differ, and each remaining difference has an explanation in the notes:

1. **Depth ties.** When two points land at the same distance on the same pixel, `GL_POINTS` keeps the one drawn first. The compute path keeps the lower **pool index**. The two orders disagree because pool order is not scene order (Post 03 and 05). The depth buffers are bit-identical except for 3 577 pixels at fit-all and 2 112 close-up, which are the same pixels that show colour differences.
2. **Rounding in the depth value.** `GL_POINTS` stores `z/w` as the hardware computed it; the resolve stores `near / w`. These can differ in the last bit.
3. **Sub-pixel snapping.** The hardware snaps window coordinates to a fixed-point sub-pixel grid before it rounds to a pixel. The shader floors a float. A point close to a pixel edge can land one pixel over, in a different image. After the checker fix, 91 (fit) and 1 774 (close) of the differing pixels are true coverage differences, and they are symmetric one-pixel shifts: about 78 % of them sit right next to the other image's pixel.

On the third item the project tried to match the hardware: emulate the 8-bit sub-pixel snapping in the shader. That **doubled** the number of differing pixels, so it was not done. The recorded decision is to accept the one-pixel shifts. That is the right call as long as you write it down with the measurement, so the next person does not repeat the experiment.

The tint (the per-cloud colour mix) matched too: 3 413 differing pixels against 3 418 without a tint, i.e. the same ties. And forcing the pool into tiny 100 MB chunks (`PCE_MAX_BLOCK`, Post 03) gives **byte-identical** images, which is the check for the chunking code.

For wide points the differences move: size 3 fit 2 295 / 46 coverage, size 7 fit 1 631 / 24. Coverage differences shrink as the point grows because a one-pixel shift of a 7-pixel square is mostly covered anyway; close-up the total grows (size 7: 7 019 differing, 875 coverage), plausibly because large near points make depth ties more likely (my reading; the notes do not break it down).

## Make the harness fail

A comparison that always prints a number is a log, not a test. The project adds explicit FAIL conditions for the paths that should be exact:

- **Block culling must draw nothing different** (Post 15). The harness renders the compute path with and without culling and requires `0` differing pixels, else it prints `FAIL`.
- **The mesh path must match to the channel** (Post 16). Any coverage difference, or a colour channel off by more than 1, prints `FAIL`.

These checks run in the `--bench` harness at the window's default 1600 x 900, so their totals are out of 1 440 000 pixels.

A harness only counts if it has caught something. So the culling work was **mutation-tested**: break the shader on purpose and confirm the check notices.

| deliberate mutant (close view) | pixels that differ |
|---|---|
| cull a block when **any** corner is outside instead of all | 130 221 |
| block index off by one | 655 116 |
| remove the rounding margin in the plane test | 0 (not visible on these views) |

The last row is instructive. The harness is *not* sensitive to the rounding margin on these views, so that behaviour is guarded by a separate CPU test that constructs a box exactly on a clip plane (132 cases where a margin-free test drops a drawn point). Know what your image diff cannot see, and put a unit test next to the thing it cannot see.

## Find the cause by removing one fix at a time

When the mesh renderer (Post 16) first appeared, it differed from `GL_POINTS` in 0.01-0.6 % of pixels (worst: close-up intensity colouring, 8 919 pixels). There were three suspected fixes. The project isolated each with a variant that had only that fix removed, and found each one mattered:

1. an edge guard on the per-pixel dedup,
2. clipping against the point's footprint instead of its centre,
3. computing colour per fragment instead of packing it as `packUnorm4x8`, which rounds differently from the RGBA8 target.

After all three the mesh path differed in **0 of 1 440 000 pixels**, largest channel difference 0, in 20 steps (fit, close and ortho; sizes 1 to 7; four colour modes; EDL; grid). One residual: on one tile alone, close-up elevation had 2 pixels off by 1 LSB (162 vs 163), because `GL_POINTS` interpolates its colour varying and perspective correction can move a value across a rounding boundary. A variant that computes colour in the mesh shader and interpolates it the same way gives 0 there, so the cause was confirmed, then left alone.

## A checklist for your own parity harness

1. **Keep a slow reference path** and render both from the same camera, every build.
2. **Split differences into coverage vs colour.** Coverage is a missing or extra point; colour is ties and rounding.
3. **Compare at the simplest setting first** (size 1, RGB), then widen.
4. **Check your checker.** Use exactly representable clear colours, and write a mutant that proves the diff can fail.
5. **Record decisions with numbers**, including the experiments you rejected.
6. **Define exactness per path.** Culling must be 0; a different rasterizer may be 0.1 %.
7. **Add unit tests where images are blind**, such as rounding margins and chunk boundaries.

Next: the first feature that has to match `GL_POINTS` pixel for pixel, point size: [Post 12](../post12-point-size-without-sprites/).

## Where this lives in the code

`src/app/bench/bench_frame.cpp` (`bench_step_report`), `src/app/bench/bench_util.cpp` (`image_diff`, `max_channel_diff`), `tests/render_model/test_render_model.cpp`. Numbers: ROADMAP 3.2 (the correction note) and 3.3b, BENCH.md "R3" and "Mesh shader pipeline". Commit 9976b1a fixed the background colour.
