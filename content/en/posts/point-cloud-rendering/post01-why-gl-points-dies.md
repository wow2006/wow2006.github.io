---
title: "Post 01: Why GL_POINTS Dies at 78 M Points"
date: 2026-10-14T10:00:00+03:00
weight: 2
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - benchmarking
---

The first thing everybody tries is `glDrawArrays(GL_POINTS, 0, n)`. It is three lines, it works on a laptop, and it looks fine on a 1 M point scan. This post shows what it costs on 78 M points, why, and how to measure it so the number can be trusted. Every later post in the series is a reaction to this one.

## The setup

| | |
|---|---|
| GPU | RTX 4090, driver 616.92, clocks not locked (2715-2730 MHz, P0, sampled with `nvidia-smi`) |
| Data | Toronto-3D, 4 PLY tiles, 78 320 210 points |
| Framebuffer | 2400x1350 (3.24 M pixels) |
| Build | RelWithDebInfo, vsync off |

The baseline is as plain as it gets: one position buffer, one colour buffer, a vertex shader that applies `proj * view * model`, and one draw call per cloud. Point size is 1 px.

## The number

Median GPU time of the scene pass over 600 frames, point size 1:

| view | GL_POINTS | compute rasterizer (post 9) |
|---|---|---|
| fit all (every point on screen) | 23.6 ms | 1.24 ms |
| close up (inside the street) | 2.43 ms | 1.27 ms |

At fit-all, 23.6 ms means 42 fps for a viewport that does nothing but orbit. And that is *before* the editor draws a selection, an overlay or a UI. The target for a navigable editor is a frame you can spend on other things, so this is already over budget.

Two things in that table are worth staring at.

**Close-up is fine.** Inside the street, most points are outside the frustum, the vertex shader still runs for all of them, and the whole thing costs 2.4 ms. `GL_POINTS` is not slow in general, it is slow when the points are all *visible*.

**The cost does not come from memory bandwidth.** Position (12 B) plus colour (4 B) for 78.3 M points is about 1.25 GB per frame. Over 23 ms that is roughly 54 GB/s, a small fraction of what a 4090 can move. If bandwidth were the limit, close-up would be nearly as slow as fit-all, because the vertex shader reads every point either way. (This is an inference from arithmetic, not from a profiler capture. Post 29 covers the profiler side.)

## Why: 24 points per pixel

78.3 M points over 3.24 M pixels is about 24 points per pixel at fit-all. The hardware rasterizer is built for triangles that cover many pixels. For a 1-pixel point it does the full setup, coverage and depth-test work to produce one fragment, and then 23 of every 24 of those fragments lose the depth test. All that work was spent on points nobody sees.

The size sweep supports this. Fit-all `GL_POINTS` time as the point gets bigger:

| point size | GL_POINTS (fit) | compute (fit) |
|---|---|---|
| 1 px | 23.2 ms | 1.25 ms |
| 3 px | 28.3 ms | 1.23 ms |
| 5 px | 43.3 ms | 1.28 ms |
| 7 px | 68.2 ms | 1.37 ms |

`GL_POINTS` pays per covered pixel, so the cost grows with the *area* of each point, 49 pixels at size 7. The compute path barely moves. That asymmetry is the whole reason the series exists.

The same pattern shows up in the fly-through. Along a camera path that circles the scene and dives into it, `GL_POINTS` has a median of 9.3 ms but a p99 of 26 ms, because the path alternates between the cheap close view and the expensive fit-all view. Compute stays at 1.26 / 1.34 ms (p50 / p99). A renderer you navigate with has to be judged by p99, not by the median of a friendly view.

## Measuring it honestly

The 23.6 ms above only means something because of how it was taken. The rules the project settled on:

**1. Time the GPU, not the CPU.** `glDrawArrays` returns in microseconds. Wall-clock around the call measures how fast the driver queues work. Use a `GL_TIME_ELAPSED` query around the pass you care about.

**2. Never stall on a query.** Reading a query result the same frame blocks the pipeline and distorts the very thing you measure. Use a ring of queries and read the oldest one that is ready:

```cpp
// N slots; never re-begin a slot whose result has not been read.
struct GpuTimer {
    static constexpr int N = 2 + 2;                 // frames in flight + 2
    GLuint q[N]{};
    bool pending[N]{}, active = false;
    uint64_t frame = 0;
    double ms = 0.0;
};

void gpu_timer_begin(GpuTimer& t)
{
    for (int k = 0; k < GpuTimer::N; ++k) {
        int i = (t.frame + k) % GpuTimer::N;
        if (!t.pending[i]) continue;
        GLint ready = 0;
        glGetQueryObjectiv(t.q[i], GL_QUERY_RESULT_AVAILABLE, &ready);
        if (!ready) continue;                       // not done yet: try again next frame
        GLuint64 ns;
        glGetQueryObjectui64v(t.q[i], GL_QUERY_RESULT, &ns);
        t.ms = ns * 1e-6;
        t.pending[i] = false;
    }
    int slot = t.frame % GpuTimer::N;
    t.active = !t.pending[slot];                    // skip the sample if the slot is still in flight
    if (t.active) {
        glBeginQuery(GL_TIME_ELAPSED, t.q[slot]);
        t.pending[slot] = true;
    }
}

void gpu_timer_end(GpuTimer& t)
{
    if (t.active)                                   // only end a query this frame actually began
        glEndQuery(GL_TIME_ELAPSED);
    ++t.frame;
}
```

The result is a few frames old, which is fine for a statistic. The `active` flag matters: when a slot is still in flight the frame takes no sample, and calling `glEndQuery` without a matching begin is a GL error. The editor sizes the ring to the frames in flight plus two.

**3. Limit frames in flight.** Swap alone does not throttle on every driver: on NVIDIA under Wayland the CPU ran to 399 frames while the GPU had finished 140, about 5 seconds of lag. The renderer allows at most 2 frames in flight with a `glFenceSync` after the swap and a `glClientWaitSync` at the top of the next frame. Without it, the CPU frame time you read is meaningless.

**4. Turn vsync off and warm up.** The bench discards 30 to 60 warm-up frames, then takes the median of 300 to 600. Shader compilation, first-draw driver work and buffer residency all land in the first frames.

**5. Log the clocks.** An unlocked GPU changes P-state under load. Early runs saw the same scene go from 4.3 ms to 7.6 ms when the card dropped state. The bench ran `nvidia-smi -lms 250` alongside and recorded the clock range in the results table. If you cannot lock clocks (no admin rights here), at least write down what they were.

**6. Report p50, p99 and max, and say which view.** The "18-20x faster" headline from the first compute-rasterizer measurement turned out to hold only for the fit-all view. Close-up the gap is 1.9x. A later pass re-checked the claim on a different driver and OS and wrote the nuance next to the number. Keep a table of every measurement with its commit hash, and add rows rather than overwriting them, so before and after stay visible.

**7. Pin the camera path.** The bench drives the camera with a frame-indexed path (`fit`, `close`, `fly`), not wall-clock time, so two runs render exactly the same frames.

## What you should take from this

- `GL_POINTS` is a fine first renderer and a bad last one. Its cost scales with the number of *visible points times covered pixels*, and at 24 points per pixel you are paying for the 23 that lose.
- Culling in the vertex shader does not help much: the vertex shader still runs, the cost you see is in primitive setup and rasterization.
- The fix is not a faster draw call, it is a different pipeline. Do the per-pixel decision yourself, in a compute shader, with one atomic per point.

Next up, before the compute rasterizer, we need the data to be arranged well enough for the GPU to read: [Post 02: Data layout for GPU point clouds](../post02-data-layout-for-gpu-point-clouds/).

## Reproduce

```bash
pce --bench-path fit --renderer compute --frames 600 --size 1 --res 2400x1350 L001.ply L002.ply L003.ply L004.ply
pce --bench-path fit --renderer points  --frames 600 --size 1 --res 2400x1350 L001.ply L002.ply L003.ply L004.ply
```

All numbers come from the project's `docs/BENCH.md`, section "F0.1 — renderers", commit `25c3bb9`, and the compute-rasterizer and point-size tables in `docs/ROADMAP.md` (3.2, 3.3b).
