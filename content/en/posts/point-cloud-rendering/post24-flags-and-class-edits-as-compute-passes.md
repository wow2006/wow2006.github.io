---
title: "Post 24: Flags and Class Edits as Compute Passes"
date: 2026-10-15T18:00:00+03:00
weight: 25
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
  - editing
---

After [Post 23](../post23-gpu-selection-and-picking/) a point can be selected. This post is about what the editor does with a selection: hide, delete, crop, reclassify. They are all the same kind of operation, a compute pass over a byte-per-point array, and the interesting design questions are where the truth lives, how to write a byte array from 128 parallel threads without a storm of atomics, and how to keep the render passes from paying for edits that are not there.

## One byte of state per point

```cpp
enum Flag : uint8_t { kSelected = 1, kHidden = 2, kDeleted = 4 };
```

Each point has a flags byte and a class byte, both in the pool ([Post 03](../post03-one-big-pool/)). Everything an editing tool does is a transformation of those two bytes:

| op | what it does to a point |
|---|---|
| select / select-all / clear / invert | sets or clears `S` on live points |
| hide selected | sets `H`, clears `S` |
| unhide all | clears `H` |
| delete selected | sets `D`, clears `S` |
| crop | sets `D` on live points that are *not* selected |
| assign class | writes the class of selected points |
| select class, select by mask, set classes | selection or class writes from other sources |

A point is **live** if it is neither hidden nor deleted; ops act on live points. Hide and delete also deselect, a rule recorded in the code comment: a hidden selected point would otherwise be removed by "delete selected" without the user seeing it. **Delete is soft**: the point stays in memory with `D` set, is skipped by every render pass, and is dropped for real only when the points are written out: Save leaves deleted points out of the file, and Merge and Subsample drop them from their result. That is why a delete over 100 M points is instant.

## The GPU is the truth

For a regular cloud, these bytes are edited **on the GPU only**. The CPU `flags` and `cls` arrays from [Post 02](../post02-data-layout-for-gpu-point-clouds/) are stale between saves. Anything on the CPU that needs the current state (save, merge, subsample, ICP, segmentation) first calls `snapshot_edits`, which reads the two byte arrays back: about 0.45 ms per million points, 109 ms at 143 M, a documented exception to the no-freeze rule. For the streamed clouds of Part 5 the roles invert ([Post 26](../post26-edits-on-streamed-clouds/)).

## The edit kernel's trick: shared memory first

The obvious kernel does one `atomicXor` on a global word per changed point. That works and it was measured: "select all / clear / hide 1.43 / 2.00 / 3.32 ms" with global atomics per point, against 0.354 / 0.429 / 0.546 ms for the final design. (The notes caution that the first figure still included an unrelated first-clear cost, so the 2-4x gap is an estimate by subtraction, not an isolated measurement.)

The final kernel gives each workgroup of 128 points a small shared-memory accumulator, and flushes it with one atomic per touched word:

```glsl
shared uint sx[33];    // XOR per flag (or class) word, from word g0 >> 2
shared uint sp[3][5];  // delta bits per plane, from word g0 >> 5

const uint old = (flags[g >> 2] >> sh) & 0xffu;           // sh = (g & 3) * 8
...
uint f = old;  /* compute the new flags for this point from op, live, selected */
const uint d = f ^ old;
if (d != 0u) {
    atomicOr(sx[(g >> 2) - (g0 >> 2)], d << sh);          // shared-memory atomic: cheap
    ...record the delta bits per plane (Post 25)
}
barrier();
if (l < 33u && sx[l] != 0u) {
    atomicXor(flags[w], sx[l]);                           // one global atomic per word touched
}
```

Flags are packed four per `uint` (one byte per point, [Post 03](../post03-one-big-pool/)), so a 128-point workgroup touches 32 words, 33 when it straddles an alignment boundary: hence `sx[33]`. Within a workgroup the 128 threads OR their per-byte XOR masks into shared memory with cheap shared atomics, and after a barrier at most 33 threads each do a single global `atomicXor`. A point whose flags did not change contributes nothing.

Class assignment uses the same structure with a byte XOR (`new ^ old`) of the class byte, `atomicXor` into `cls` and `atomicOr` into the delta.

## Don't pay for edits that are not there

Most of the time nothing is selected, hidden or deleted. The render passes should not read the flags buffer then. The rules the project adopted, each with a measurement:

- **`any_flags`**: set by any op, cleared only when a count readback issued after the last op shows none.
- **The compute raster is compiled twice** (`PCE_FLAGS`), and the flag-free variant runs while `any_flags` is false. A uniform-gated check in one program cost 0.08 ms at 78 M points even when off ([Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/)).
- **The resolve** gates the read with a uniform (`useFlags`).
- **`GL_POINTS`** points the flags vertex stream at a 16-byte zero buffer with stride 0 when no flag is set. The obvious alternative, disabling the vertex attribute, **recompiles the vertex shader** in the driver (debug id 131218). Fetching flags cost 2.34 → 2.70 ms close-up in the draw path.

With a flag set, the cost is real but small: scene pass 1.193 ms with none → 1.327 ms with flags read (raster +0.08, resolve +0.05), the same with 60 M points selected, and 0.628 ms with 60 M points hidden because they are skipped.

Selected points are highlighted by blending 60 % toward a fixed red in every colour mode, and the tint goes on top. The notes flag a known weakness: the fixed red clashes with class colours and the top of the elevation ramp.

## Operation costs

Medians on 78.3 M points, 60 M selected where relevant, RTX 4090, including the delta clear:

| op | ms |
|---|---|
| select all | 0.354 |
| clear selection | 0.429 |
| invert | 0.465 |
| hide | 0.546 |
| unhide | 0.433 |
| delete | 0.553 |
| crop | 0.357 |
| assign class (byte XOR, plus a 75 MB delta) | 0.739 |

Each is far inside a frame. The bench drives them through ImGui's event queue, exactly like a user, and compares the GPU's flags and classes against a CPU mirror after each op.

## Classes that hide

The class panel can hide a class, which means its points are neither rendered nor edited. The kernels treat `classHiddenAt(k)` points as not live and not selected via a small class mask uniform, exactly like a hidden flag, with no change to the flag bytes. The point counts per class come from a separate counting pass (`count_classes`) into the same small stats buffer as the selected, hidden and deleted counts.

## Checklist

1. **Represent edit state as a few bytes per point** and make every tool a transformation of them.
2. **Soft-delete**, compact on save: edits stay O(passes), not O(data movement).
3. **One writer for the truth** (the GPU), with an explicit, measured snapshot for anything on the CPU.
4. **Accumulate in shared memory**, flush with one atomic per word.
5. **Compile the no-edit path separately**; measure what a gate costs.
6. Verify every op against a **CPU mirror** through the real UI path.

Next: undo for a byte array that lives on the GPU and has 78 M entries: [Post 25](../post25-undo-redo-as-gpu-xor-deltas/).

## Where this lives in the code

`src/render/render.cpp` (`kEdit`, `kCount`, `edit_points`, `count_flags`, `count_classes`), `src/render_model/render_model.hpp` (`EditOp`, `kOpPlanes`, CPU references), `src/app/scene/edit_ops.cpp` (`run_op`, `any_flags`). Numbers: ROADMAP 4.1, 4.6, 4.8, BENCH.md "Task 2.3". Commit 694e40c.
