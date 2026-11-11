---
title: "Post 17: Choosing a Renderer at Runtime"
date: 2026-10-15T11:00:00+03:00
weight: 18
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - architecture
---

The series now has three ways to draw the same points: the compute rasterizer ([Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/)), the mesh-shader pipeline ([Post 16](../post16-mesh-shaders-for-points/)), and plain `GL_POINTS`. This post is about the unglamorous part that makes that safe: deciding which one a frame uses, deciding it in one tested function, and telling the user *why*.

## Why more than one renderer

Each path fails on different inputs:

| path | needs | limit | strength |
|---|---|---|---|
| compute | `GL_ARB_gpu_shader_int64`, `GL_NV_shader_atomic_int64`, the shaders compiled | the resolve binds `rgba` (and `pos` for elevation) in at most 4 chunks, about 715 M points | fastest at fit-all; owns `fb64` for picking and edits |
| mesh | `GL_NV_mesh_shader` and its programs | opt-in; no `fb64` | fastest close-up on the measured laptop GPU |
| `GL_POINTS` | nothing special | slow at fit-all (23.6 ms) | always works |

A new feature should never require a particular row to exist. A driver without 64-bit atomics, a shader that fails to compile on someone's machine, or a 900 M-point scene must degrade to something that still draws.

## One pure function decides

The decision is a `constexpr` function in the GL-free header, with a result type that carries the reason:

```cpp
enum class RendererReason { kCompute, kMesh, kUnsupported, kUserOff, kPoolTooBig };
struct RendererChoice { bool compute; RendererReason reason; };

constexpr RendererChoice choose_renderer(bool compute_ok, bool use_compute, const ResolveChunks& rc,
                                         bool mesh_ok = false, bool use_mesh = false)
{
    if (mesh_active(mesh_ok, use_mesh))   return {false, RendererReason::kMesh};
    if (!compute_ok)                       return {false, RendererReason::kUnsupported};
    if (!use_compute)                      return {false, RendererReason::kUserOff};
    if (!rc.fits)                          return {false, RendererReason::kPoolTooBig};
    return {true, RendererReason::kCompute};
}
```

Read the order as a priority list:

1. **Mesh wins only if the user asked for it and it exists.** The mesh path is opt-in (Post 16: slower at fit-all, no pick buffer), so a machine that supports it still runs compute by default.
2. **Compute unavailable** is reported as `kUnsupported`, even if the user also unticked compute.
3. **The user switched compute off**: the reason shown is the user's choice (`kUserOff`). A big pool and an unticked checkbox report the checkbox, since that is the cause the user can change.
4. **The pool does not fit the resolve's chunks** (`ResolveChunks.fits`, from Post 03): `kPoolTooBig`.
5. Otherwise compute.

Five inputs, all computed elsewhere, and nothing in the function touches GL. That is the design rule from the project's notes: anything the LOD code or the tests need from the renderer lives in a header with no GL types.

## The reason is a string the user sees

```cpp
constexpr const char* renderer_label(RendererReason r)
{
    switch (r) {
    case RendererReason::kCompute:     return "compute raster";
    case RendererReason::kMesh:        return "mesh shader";
    case RendererReason::kUnsupported: return "GL_POINTS (compute raster unsupported)";
    case RendererReason::kUserOff:     return "GL_POINTS (Compute raster unchecked)";
    case RendererReason::kPoolTooBig:  return "GL_POINTS (pool over 4 chunks of max block)";
    }
    return "";
}
```

The Stats panel shows it, and `--bench-path` prints it, so a benchmark that silently fell back to `GL_POINTS` cannot be mistaken for a compute result. Without that, "why is this scene slow on the customer's machine?" costs a debugging session; with it, the answer is on screen. The reason display was added as task F4.5 ("show the renderer in use, and why").

## Tests that pin the contract

Three Catch2 tests cover the function with no GL context:

- `mesh shader is opt-in, compute stays the default`: mesh available but not ticked gives compute; ticked and available gives mesh; ticked but missing falls back; mesh works even without compute; default arguments mean no mesh.
- `choose_renderer reasons and labels`: with the real numbers (78 M points fit, 720 M do not), every combination of availability, user choice and pool size yields the expected reason; every reason has a label, labels are unique, and their prefix names the renderer.
- `unticking mesh returns to compute`: the UI state machine (tick, untick, switch compute off, mesh unavailable) leaves the user's compute setting untouched, so unticking mesh puts you back where you were.

The last one is a real bug class: a checkbox that changes two booleans, so toggling it back does not restore the first state. Making the mesh opt-in its own flag (and `mesh_active` a pure function of two flags) keeps compute's setting independent.

## Capability discovery at startup

Everything above relies on what init discovered:

```cpp
mesh_ok    = mesh && mesh_lod;                        // both mesh programs linked
compute_ok = raster && raster_flags && raster_lod && raster_lod_flags && resolve;
cull_ok    = compute_ok && raster_cull && raster_cull_flags;   // R3: no cull programs, then everything draws unculled
```

Each program is compiled and linked; a failure prints the driver's log to stderr and leaves that program empty, instead of aborting. The startup log prints the extension checks and the limit that sets the chunking:

```
GL_ARB_gpu_shader_int64:   yes
GL_NV_shader_atomic_int64: yes
GL_NV_mesh_shader:         yes
GL_MAX_SHADER_STORAGE_BLOCK_SIZE: ... B -> 178 M points of pos per SSBO binding
```

and if compute is unavailable it says what is lost, in plain words: *"compute rasterizer unavailable: using GL_POINTS (no picking, no visible-only selection)"*, or, with mesh available, that drawing works but picking and visible-only selection do not. Degraded features are named, not hidden.

Block culling degrades separately: without the cull programs everything draws unculled, and `PCE_NO_CULL=1` is the A/B switch. Environment variables are the project's test hooks: `PCE_MAX_BLOCK` lowers the SSBO limit so chunked paths run on small data, `PCE_GL_POINTS` forces the fallback for LOD benchmarks.

## What the fallback costs you

`GL_POINTS` is not just slower; it is *less capable*. Without the compute rasterizer there is no `fb64`, so there is no pixel-exact picking and no selection restricted to visible points. In `GL_POINTS` mode (compute switched off but available) the pick pass rasters `fb64` on demand, a path the project's own notes list as untested; where compute is unsupported there is no picking at all, as the startup message says. When you plan fallbacks, list what each one loses, then say it in the UI and the log.

## Checklist

1. **One pure function** turns capabilities, user choices and scene size into a choice.
2. Return a **reason**, not just a boolean, and show it in the UI and in benchmark output.
3. **Order the rules by who can fix the cause**: the user's own choice before a limit they can't change.
4. Keep every **user setting independent**, so toggling restores the prior state.
5. **Compile at startup, degrade per program**, log the driver errors, and name the lost features.
6. Put **test hooks** (env vars) at every limit so the unusual paths run on small data.

That ends Part 4. Part 5 leaves the single-resident-pool model behind: how to stream clouds bigger than VRAM, starting with the octree on disk: [Post 18](../post18-out-of-core-octree/).

## Where this lives in the code

`src/render_model/render_model.hpp` (`choose_renderer`, `RendererReason`, `renderer_label`, `mesh_active`), `src/app/init.cpp` (program creation, `compute_ok`, `mesh_ok`, `cull_ok`), `src/app/gl/gl_info.cpp` (extension and limit queries), `tests/render_model/test_render_model.cpp`. Commits: 4e3f4da (show the renderer in use, and why), d473440 (mesh opt-in), 98f3b90 (unticking mesh returns to compute).
