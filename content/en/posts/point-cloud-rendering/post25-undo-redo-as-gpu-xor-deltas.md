---
title: "Post 25: Undo/Redo as GPU XOR Deltas"
date: 2026-10-15T19:00:00+03:00
weight: 26
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - undo
---

An editor without undo is a trap, and an editor whose undo copies 78 M bytes through the CPU is slow. The obvious design is a CPU snapshot of the changed points. The roadmap even planned that, and then deliberately deviated: "every edit records `new ^ old` on the GPU". This post covers why XOR is the right primitive, how the deltas are laid out, how the history stays within VRAM, and how it survives the pool being compacted underneath it.

## Why XOR

For a byte `b` that an op changes to `b'`, define `d = b ^ b'`. Then `b' = b ^ d` and `b = b' ^ d`. **The same delta undoes and redoes.** There is no "before" and "after" copy, no direction, and the apply kernel is the same for both:

```glsl
// Undo and redo: dst ^= delta, one thread per 4-point word (no atomics: each word has one owner).
void main() {
    const uint w = gl_GlobalInvocationID.x;
    if (w >= nwords) return;
    uint x = 0u;
    if (planes == 0u) x = delta[w];                       // class: the byte XOR itself
    for (uint b = 0u, k = 0u; b < 3u; ++b)
        if ((planes >> b & 1u) != 0u) {
            const uint q = (delta[k * words + (w >> 3)] >> ((w & 7u) * 4u)) & 15u;   // this word's 4 points
            x |= ((q & 1u) | (q & 2u) << 7 | (q & 4u) << 14 | (q & 8u) << 21) << b; // point j -> byte j, bit b
            ++k;
        }
    if (x != 0u) dst[w] ^= x;
}
```

There are no atomics: each output word (4 points) has exactly one thread. And because XOR is commutative and self-inverse, the order of applying independent deltas to different bits does not matter, and applying a delta twice returns to the start (50 undo/redo cycles give identical state, the roadmap's done-when).

## Bit-planes for flags, bytes for classes

A delta is recorded by the same edit kernel that applies the op ([Post 24](../post24-flags-and-class-edits-as-compute-passes/)): when a point's flags change by `d`, the kernel sets the matching bit in a per-plane bit array.

- **Flags** are recorded as **bit-planes**: one plane per flag bit that the op *may* change (`kOpPlanes[op]`). Select changes only `S`: one plane. Hide changes `S` and `H`: two. A plane is 1 bit per point, **9.3 MB at 78 M points** (the same as `n / 8`), where storing flag bytes would take 75 MB.
- **Classes** are a **byte XOR**: 75 MB at 78 M points, since the byte can change in all eight bits. That is the largest delta, so assign class is the most expensive undo step.

The planes a given op needs are a compile-time table; for example select-all and invert record `kSelected`; hide records `kSelected | kHidden`; delete records `kSelected | kDeleted`; crop only `kDeleted`; assign class records none (it uses the byte form).

Apply-time cost, medians on 78.3 M points: **0.08-0.15 ms for flag ops, 0.20-0.21 ms for a class op.** An undo is cheaper than the op that created it, since there is no decision logic.

## The history lives in VRAM

```cpp
struct Edit {
    gl::GLuint buf = 0;                 // the delta, on the GPU
    std::size_t n = 0, bytes = 0;
    uint8_t planes = 0;
    uint64_t cloud = 0;                 // with no buffer: this cloud's transform goes from `before` to `after`
    Transform before, after;
    std::vector<uint64_t> lod;          // LOD clouds whose op log took this op (Post 26)
};
struct History { std::vector<Edit> undo, redo; std::size_t bytes = 0, cap = ...; };
```

The deltas are device buffers, never copied to the CPU. The cost is VRAM, so the history needs a cap:

- **The cap is derived from the card**: an eighth of total VRAM, clamped to [256 MB, 8 GB]; on the 24 GB card 3 GB, "~300 selection ops on 78 M points" (a flag plane is 9.3 MB). It was a fixed 2 GB before the fix. `history_cap()` is a one-line `constexpr` function with a test.
- **The oldest steps are dropped first**, and one step is always kept. `run_op` also drops older steps when the *free* VRAM (from the NVX extension) minus a margin cannot hold the new delta.
- **The Edit panel shows usage** ("undo / redo, MB of cap MB"), because a limit that the user cannot see is a surprise.
- A new edit **drops the redo stack**, as in any editor.

A bench stage forces the cap path (it had never run): with the cap set to 2.5 deltas, four selects keep two steps within the cap, and undoing them lands on the state after the second select.

## Transforms are history too

Moving a cloud with the gizmo is also an undoable action but needs no delta: an `Edit` with a `cloud` id and no buffer stores the `before` and `after` transforms. The undo stack is one list in timeline order, mixing both kinds.

## Removing a cloud must not erase history

Remove compacts the pool ([Post 03](../post03-one-big-pool/)): surviving ranges move, so a delta indexed by pool position would now point at the wrong points. The first implementation simply cleared the whole history on Remove, a data-loss surprise. The fix remaps the deltas through the same moves, on the GPU:

```cpp
void history_remap(const EditPrograms& p, const GpuPool& pool, gl::GLuint program, History& h,
                   std::span<const PoolMove> moves, std::size_t n_new, std::span<const uint64_t> removed);
```

A `kRemap` compute pass (CPU reference `delta_remap`) rewrites each delta in the new indexing, for both planes and class bytes. Entries left with nothing of their own (the removed cloud's transform, LOD ops of removed LOD clouds only) are dropped. It works because **delta indices are stable otherwise**: clouds append to the pool, growth keeps offsets, and only compaction moves things.

Merge and subsample still clear the history (they replace their source clouds; the UI says so). The bench removes a cloud, undoes twice to reach the start state and redoes twice, for all 68 M kept points.

## The fresh-buffer trap again

A delta is a fresh buffer that is cleared before first use. On the NVIDIA driver the first write to a never-drawn buffer ran at about 20 GB/s ([Post 03](../post03-one-big-pool/)'s placement trap). `edit_alloc` therefore calls `first_draw` on the new buffer before its first write: a 9 MB delta clear went from 0.69 ms to **0.023 ms**, and ops from 1.1-4.4 ms to 0.35-0.74 ms.

## How it was verified

- A CPU reference `delta_record` / `delta_apply` with a round-trip test, where each deliberate break is detected (mutation-tested).
- Through the bench: 13 ops undone gives flags and classes byte-identical to the start and an image with 0 pixels different; redone, byte-identical to the end state; 55 op/undo/redo/undo cycles for each of the 11 timed ops return to the start.
- `delta_remap` after compaction (planes and class bytes, clouds added after the edit), mutation-tested.

## Checklist

1. Pick a delta that is **its own inverse** (XOR), so undo and redo are one kernel.
2. Record it **in the op's own kernel**, as the only extra output.
3. Use **bit-planes** for flags and **bytes** for classes; size by what an op can change.
4. **Cap history by VRAM** (derived, not fixed), drop oldest, show usage.
5. **Remap, don't clear**, when storage is compacted; test the remap.
6. Allocate fresh GPU buffers with the **placement workaround** if your driver needs it.

Next: the same operations on a cloud that is mostly not in memory: [Post 26](../post26-edits-on-streamed-clouds/).

## Where this lives in the code

`src/render/render.cpp` (`kXor`, `kRemap`, `edit_alloc`, `edit_finish`, `xor_apply`, `history_remap`), `src/render_model/render_model.hpp` (`delta_record`, `delta_apply`, `delta_remap`, `history_cap`, `kOpPlanes`), tests in `tests/render_model/`. Numbers: ROADMAP 4.7, FIXES F4.2-F4.3.
