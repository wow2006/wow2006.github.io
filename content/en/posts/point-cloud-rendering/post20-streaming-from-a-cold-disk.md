---
title: "Post 20: Streaming LOD from a Cold Disk 12x Faster"
date: 2026-10-15T14:00:00+03:00
weight: 21
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - io
  - benchmarking
---

The numbers in [Post 18](../post18-out-of-core-octree/) were taken with the `.pco` file in the OS page cache. A user who opens a 24 GB file for the first time does not have that. This post is the story of measuring the **cold** case, finding the cause, and fixing it with two small changes. It is also a case study in letting a measurement decide whether a fix is needed at all.

## Step 0: measure before fixing

The fix list had a conditional item: "streaming throughput, *conditional on* a cold-disk measurement". The rule written next to it: if cold MB/s is under 50 % of what the drive can do at queue depth 32, do the fix.

On Windows there is no non-admin way to flush the file cache, so a cold run reads a **fresh copy made with `robocopy /J`** (unbuffered I/O: the copy has no pages in the cache). A second run on the same copy is the warm one. The drive was measured independently (`pce --bench-disk`, 8 GB unbuffered, independent of the cache):

| drive test | MB/s |
|---|---|
| sequential 1 MB, queue depth 32, 1 thread | 7016 / 6968 |
| queue depth 1, 1 thread | 3550 / 3741 |
| sequential mapped read, 1 thread, cold | 1373 |

Then the streaming bench, heavy configuration (budget 100 M points, detail 0.5 px, 2400 x 1350, 30 582 pages = 125 M points, 2.4 GB of cache):

| view | MB | cold: full detail | cold MB/s | warm: full detail | warm MB/s |
|---|---|---|---|---|---|
| fit all | 67 | 0.46 s | 147 | 0.06 s | 1198 |
| aerial 300 m | 180 | 0.82 s | 219 | 0.09 s | 2039 |
| street 60 m | 196 | 0.82 s | 240 | 0.09 s | 2099 |
| street 15 m | 84 | 0.36 s | 232 | 0.04 s | 1942 |

**Cold streaming ran at 147-240 MB/s: 2-3 % of the drive's 7 GB/s, and 11-17 % of a cold sequential mapped read.** Warm, the same work ran at 1.2-2.1 GB/s (the uploader's own copy rate 2.2-2.4 GB/s) and reached full detail about 9x sooner. The decision rule fired.

## Diagnosis

The uploader thread ([Post 06](../post06-persistent-mapped-staging-ring/)) `memcpy`s each node's bytes from the **memory-mapped file** into the staging slot. On a cold file, the first touch of each page is a page fault that waits for the disk. One thread faults one cluster at a time: the effective **queue depth is 1**, reading 4 to 48 KB pieces of nodes scattered over a 24 GB file. The log confirmed it: the uploader's own throughput (MB it copied divided by time inside the copy) equalled wall MB/s, so it was busy the whole settle, waiting on faults.

The hypothesis was written down before testing ("one uploader thread memcpy-ing from the mmap blocks on each page fault on a cold disk"), and the measurement matched it. That matters: it tells you which of the three candidate fixes to try first.

## Fix 1: tell the OS what is coming

Each frame the selection knows which nodes are about to be loaded. Before the uploader touches them, ask the OS to read all of them in at once, with large concurrent reads:

```cpp
// lod_frame: one range per node issued this frame, from its first byte to the end of its last array
c.prefetched.clear();
if (c.prefetch && !issue.empty()) {
    for (uint32_t i : issue) {
        const OctNode& on = o.nodes[i];
        const Octree::Layout l = o.layout(on);
        const std::size_t end = std::max({l.color + l.color_stride * on.count, l.intensity + 2 * std::size_t(on.count),
                                          l.cls + on.count});   // v1 ends with cls, v2 with the colours
        c.prefetched.push_back({o.data + on.offset, end - on.offset});
    }
    prefetch(c.prefetched);                                     // one call for the whole frame
}
```

`prefetch` is `PrefetchVirtualMemory` on Windows and `madvise(MADV_WILLNEED)` elsewhere (`src/core/mmap.hpp`). Result on v2 cold: time to full detail dropped from about 0.49 / 0.51 / 0.22 s to **0.13 / 0.14 / 0.06 s**, i.e. cold became equal to warm.

A test guards the contract: **every task's file bytes must lie inside the frame's prefetched ranges**. It caught a real gap on the first run: v1 nodes' intensity and class arrays were outside the ranges computed from the position layout, so those pages still faulted one at a time.

## Fix 2: more than one thread per slot

After fix 1 the cold time equalled warm but warm itself was now bound by one thread decoding ([Post 19](../post19-quantized-nodes/)). The staging slot is filled by `parallel_for` with a small worker count:

```cpp
parallel_for(batch.size(), 8, [&](std::size_t b, std::size_t e, std::size_t) {
    for (std::size_t k = b; k < e; ++k)
        upload_fill(batch[k], dst + offs[k]);     // memcpy or v2 decode, at its packed offset in the slot
}, s.threads);                                     // Staging::threads, default 4
```

Tasks are packed back to back, so each can be filled independently. Sweep (v2, cold, seconds to full detail for aerial / street 60 m / street 15 m):

| fill threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| time | 0.13 / 0.14 / 0.06 | 0.09 / 0.10 / 0.05 | **0.07 / 0.07 / 0.04** | 0.06 / 0.07 / 0.04 |

Four threads kept; eight was not worth the cores. The in-flight budget (`kLodInflight`, at most 192 MB of uploads queued, the staging ring's size) was swept at 96 / 192 / 384 MB with the same result inside noise, so it stayed at 192.

## Result

| build | layout | cold (aerial / 60 m / 15 m) | warm |
|---|---|---|---|
| baseline: 1 uploader thread, mmap faults | v1 | 0.82 / 0.82 / 0.36 s | 0.09 / 0.09 / 0.04 s |
| + v2 | v2 | ~0.49 / 0.51 / 0.22 s | ~0.12 / 0.13 / 0.06 s |
| + prefetch | v2 | 0.13 / 0.14 / 0.06 s | 0.12 / 0.13 / 0.06 s |
| + 4 fill threads (final) | v2 | **0.06 / 0.07 / 0.04 s** | 0.11 / 0.06 / 0.03 s |
| final | v1 | 0.07 / 0.08 / 0.04 s | 0.06 / 0.06 / 0.04 s |

Cold time to full detail fell from 0.82 s to 0.07 s: **about 12x**, and cold now equals warm. The third candidate, overlapped `ReadFile` straight into the staging ring, was **skipped**: nothing was left for it to win. Notably, the file rate at the end is only 19 to 25 % of the drive's queue-depth-32 peak (1.3-1.8 GB/s). The settle is now paced by the frame loop, which pumps the three 64 MB staging slots once per frame, not by I/O.

A side effect, recorded: the CPU frame p99 while settling rose from 1.2-1.4 ms to 7-11 ms, since uploads now land every frame instead of trickling. That is the price of arriving ten times sooner.

## Lessons

1. **Write the decision rule before you measure** ("do it if cold < 50 % of the drive"). It stops you from building the third fix because it was in the plan.
2. For cold-cache I/O, **know your method of making a cold file** (here a fresh unbuffered copy) and keep it in the benchmark's documentation.
3. **Look at the thread that does the waiting.** The uploader's busy-time equalling wall time said "serialised on page faults", not "disk is slow".
4. **Prefetch with the knowledge you have**: the selection is known a frame ahead.
5. **A fix can expose the next bottleneck.** Prefetch made cold equal warm, and warm was then the decode on one thread.
6. **Test the contract of the optimisation** (prefetched ranges cover the reads); the first run found a bug.

Next: how a streamed node's data reaches the rasterizer without changing a line of it: [Post 21](../post21-feeding-the-same-shaders-from-a-page-table/).

## Where this lives in the code

`src/lod/lod.cpp` (`lod_frame`, the prefetch block), `src/core/mmap.hpp` (`prefetch`), `src/render/upload.cpp` (`parallel_for` fill, `Staging::threads`), bench: `pce --bench-disk`, `pce --bench-lod`. Numbers: BENCH.md "F0.2" and "F3", FIXES F3.
