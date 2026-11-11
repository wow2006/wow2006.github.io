---
title: "Post 29: Profiling with Tracy and Nsight"
date: 2026-10-15T23:00:00+03:00
weight: 30
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - profiling
  - tracy
---

[Post 28](../post28-benchmarking-a-renderer-honestly/) is about measuring a number. This post is about seeing *why* it is that number: where a frame's time goes on the CPU and on the GPU, and how the editor is set up so that looking is cheap and does not change the answer. The tools are Tracy (CPU and GPU zones, memory) and Nsight Graphics or RenderDoc (frame captures), plus a handful of in-app timers.

## Two builds of every module

Profiling markers are not free. The notes record a specific cost: **each new thread's first Tracy zone costs about 30 ms**, because Tracy registers every thread on first use, and `parallel_for` ([Post 07](../post07-loading-100m-points-without-freezing/)) spawns fresh `std::jthread`s on every call. Put markers in a hot parallel path of a benchmark and you measure Tracy.

So the build system makes the choice structural. Every module is compiled in two variants from the same sources:

```cmake
# pce::<module>            markers compile to nothing (Tracy's headers only) - tests, benchmarks, tools
# pce::<module>_profiled   markers on, the Tracy client linked in              - the app (only with PCE_PROFILE)
# Never link both variants of a module into one executable.
function(pce_add_library module) ... endfunction()
```

and `pce_target_link_libraries` maps a dependency on `pce::core` to `pce::core_profiled` automatically in the profiled variant. The tests, `pce_bench`, `pce_tiles` and the other tools never carry markers. The option is `PCE_PROFILE` (on by default for the app); a `no-profile` preset strips every marker for runs where even the idle client matters. The task-2.3 stall tables were taken on the no-profile build for exactly that reason. The Tracy client runs **on demand**: it stays idle until a profiler connects.

## What is marked

Rules from the project's checklist: a new thread gets `tracy::SetThreadName`, and a new hot path gets a `ZoneScoped`.

- **CPU:** frame marks (`FrameMark`), zones for load, ingest, upload copies, selection, the LOD frame, the sort and gather. Every worker is named: `"load L003.ply"`, `"uploader"`, `"worker"`. The timeline then reads as the pipeline of [Post 07](../post07-loading-100m-points-without-freezing/) rather than a heap of anonymous threads.
- **GPU:** `TracyGpuContext` at init, `TracyGpuZone("draw points")`, `"upload copies"`, `"edit ops"`, `"imgui render"`, and `TracyGpuCollect` each frame. The GL calls are routed through a thin glbinding shim.
- **Memory:** every cloud arena ([Post 02](../post02-data-layout-for-gpu-point-clouds/)) reports `TracyAllocN(base, cap, "arena")`, so loading shows up as a staircase in the memory view and a leak is visible.

## In-app timers: Tracy's smaller sibling

Not every question deserves a profiler session, and some need numbers in the UI. The Stats panel has:

- **CPU frame ms** (wall time frame to frame) and **CPU work ms** (the same minus the fence wait and the swap), because with fence pacing ([Post 01](../post01-why-gl-points-dies/)) the first is dominated by waiting and tells you nothing about the CPU's own cost.
- **GPU "scene" and "imgui"** each with its own `GL_TIME_ELAPSED` ring (frames in flight plus two deep; a slot with an unread result is never re-begun; a stale value is shown grey instead of freezing).
- **Raster vs resolve** from `GL_TIMESTAMP` marks (`GpuStamps`, up to four marks per frame), the split quoted throughout Part 3.
- **Renderer in use and why** ([Post 17](../post17-choosing-a-renderer-at-runtime/)), cloud-range count, LOD page and node counters, undo VRAM.

They cost one query per slot and are read without stalling, so leaving them on does not distort the frame they measure.

## For the GPU: name everything

A frame capture of this renderer in Nsight Graphics shows compute dispatches with no names. The renderer adds them, at the cost of one RAII struct and a few lines per resource:

```cpp
struct DebugGroup {                       // glPushDebugGroup / glPopDebugGroup
    explicit DebugGroup(const char* name);
    ~DebugGroup();
};
void gl_label(gl::GLenum identifier, gl::GLuint name, const char* label);   // glObjectLabel
...
DebugGroup group("compute raster");
gl_label(GL_BUFFER, t.fb64, "raster fb64");
gl_label(GL_BUFFER, pool_buf, "pool pos");
```

Passes are grouped as `compute raster`, `resolve`, `GL_POINTS`, `upload copies`, `compute raster (LOD pages)`; buffers read `pool pos`, `pool rgba`, `staging ring`, `cloud ranges`, `selection mask`. In a capture the frame is a tree of named passes touching named resources, which is what makes statements like "the raster reads 940 MB of `pool pos`" ([Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/)) checkable. That kind of statement, bandwidth bound or not, is exactly where this series said "an inference, not a profiler capture". The labels are what a reader would use to turn those inferences into measurements, and I have not seen such a capture in the repository.

## The driver talks: read it

OpenGL's debug output is the cheapest profiler a GL programmer has, and the editor keeps it on:

- The callback keeps **high and medium** severity and disables notifications in the driver.
- Two message ids became recurring characters in this series. **131186**, "buffer moved from VIDEO to HOST memory", is the NVIDIA placement trap ([Post 03](../post03-one-big-pool/)): it explained a roughly one-second stall per 2 GB. **131218**, "shader recompiled because of state", explained why the flags stream is pointed at a zero buffer instead of the attribute being disabled: disabling it recompiles the vertex shader ([Post 24](../post24-flags-and-class-edits-as-compute-passes/)).

Neither was found by a sampling profiler. Both were found by reading the log of a run that was slower than expected. Make "no GL debug messages" part of your bench's acceptance (the M4 benchmark states it: "no GL debug messages").

## A short diagnosis recipe

When a number is off, in this order:

1. **Is it the right renderer?** The Stats line says which, and why.
2. **CPU work or GPU?** Compare CPU work ms to GPU scene ms. A fence wait hides GPU-bound frames inside CPU frame time.
3. **Which pass?** `GL_TIMESTAMP` split (raster, resolve), or Tracy GPU zones.
4. **Any debug messages?** IDs 131186 and 131218 first.
5. **Any thread contention?** Tracy's timeline: loader threads against the uploader against the main thread. The `half the cores divided by active loaders` rule ([Post 07](../post07-loading-100m-points-without-freezing/)) came from a measured 54-75 ms CPU frame with 62 of 64 threads busy.
6. **Then open the capture**, with named passes and buffers, and look at the one dispatch that matters.

## Checklist

1. **Build markers out of anything you benchmark**; keep two variants of each module.
2. **Name every thread and every GPU pass and resource.**
3. Show **CPU work separately from frame time** when you pace with fences.
4. Keep GL **debug output on** and learn the ids your driver emits.
5. Keep cheap **in-app GPU timers** (query rings read without stalling) for the questions that do not need a profiler.
6. Turn inferences into captures whenever you can; note when you cannot.

Last post: putting it all together, with the budget of a 235-million-point frame: [Post 30](../post30-a-235m-point-frame-budget/).

## Where this lives in the code

`cmake/PceLibrary.cmake`, `cmake/Dependencies.cmake` (`PCE_PROFILE`, Tracy), `src/app/frame.cpp` (`FrameMark`, `TracyGpuZone`, `TracyGpuCollect`), `src/core/arena.hpp` (`TracyAllocN`), `src/render/render.cpp` (`GpuTimer`, `GpuStamps`, `DebugGroup`, `gl_label`), `src/app/init.cpp` (`TracyGpuContext`, debug callback). Commits bc9c090 (Tracy), dd3e4db (Nsight groups and labels).
