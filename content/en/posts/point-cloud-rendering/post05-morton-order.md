---
title: "Post 05: Morton Order and the 128-Point Block Shuffle"
date: 2026-10-14T14:00:00+03:00
weight: 6
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - morton
  - radix-sort
---

A point cloud file is in scan order: the points come out in the order the sensor measured them. That order is useless to a renderer. This post is about the one pass every cloud goes through before upload, a **Morton sort followed by a shuffle of 128-point blocks**, why a plain sort is *slower* than doing nothing, and what the 128-point block turns into later in the series.

## Step 1: Morton codes

A Morton code (Z-order curve) interleaves the bits of the quantized x, y and z coordinates into one integer. Sorting by that integer lays points out along a space-filling curve, so points that are near in 3D tend to be near in the array.

```cpp
// 3*bits-bit Morton codes: each axis quantized to `bits` bits over [bbmin, bbmax];
// x in bit 0, y in bit 1, z in bit 2.
std::vector<uint64_t> morton_codes(std::span<const glm::vec3> pos, glm::vec3 bbmin, glm::vec3 bbmax,
                                   int bits = kMortonBits);
```

The classic implementation spreads each coordinate's bits apart with a few shift-and-mask steps (`x` in bit 0, `y` in bit 1, `z` in bit 2) and ORs them together. That part is cheap. The interesting decisions are how many bits and how to sort.

### How many bits? Ten

The first version used 21 bits per axis (a 63-bit key). Measured on four Toronto tiles (78.3 M points), with the GPU "draw points" median and a 10 M-point serial sort:

| bits per axis | draw median | sort 10 M points |
|---|---|---|
| 21 | 25.45 ms | 514 ms |
| 12 | 24.32 ms | 368 ms |
| **10** | **24.01 ms** | **340 ms** |

Ten bits sorts in 3 radix passes instead of 6, and draws 5.6 % faster than 21. The reason is that only the order *within a 128-point block* matters to the renderer (next sections), and 1024 cells per axis is plenty for that. The tradeoff is documented in the code: ten bits give about 0.3 m cells on a 300 m tile, so a single km-scale file should pick its bits from the extent. That tuning has not been done, and the doc says measure before changing it. A caveat the project recorded on its own numbers: the GPU medians for the three settings were identical across two runs (25.45 / 25.45, 24.32 / 24.32, 24.01 / 24.01), which is suspicious for a noisy GPU. They were taken with the clocks not logged, and the row is marked for re-measurement. Read the bit-count result as "probably right, not nailed down".

### Sorting: a stable parallel LSD radix sort

```cpp
// Stable parallel LSD radix sort of n (key, index) pairs by the low key_bits bits of the keys,
// using (k2, i2) as scratch. The result doesn't depend on the worker count.
std::pair<uint64_t*, uint32_t*> radix_sort_pairs(uint64_t* k, uint32_t* i, uint64_t* k2, uint32_t* i2,
                                                 size_t n, int key_bits, size_t min_chunk, size_t nw,
                                                 std::stop_token stop = {});
```

The sort carries the original index next to the key and returns a **gather order**, `new[i] = old[order[i]]`, not a sorted copy. Then `apply_order` permutes every attribute (pos, rgba, intensity, class, flags) through that one order. Two details worth copying:

- **It is deterministic for any thread count.** Each pass uses digits of at most 12 bits (at most 4096 buckets, so the scatter streams stay cache-friendly): a 30-bit key is 3 passes of 10 bits, a 63-bit key 6 passes of 11. Per pass, every worker histograms its chunk, an exclusive prefix over (digit, worker) turns the histograms into per-worker offsets, and every worker scatters its chunk. Equal digits keep chunk order, so the result is the serial one for any worker count. A pass in which every key has the same digit is skipped, since it would only be a copy. The tests compare `max_threads = 1` against many threads, and the shuffle is seeded (`mt19937_64`, fixed seed), so a loaded cloud is bit-identical from run to run. This is what makes later image-diff benchmarks meaningful.
- **It is cancellable.** The stop token is polled once per radix pass and once per attribute during the gather. A stopped sort returns null, and the cloud being loaded is discarded.

Parallel speed-ups measured on the loader (median of 3, before to after parallelizing): `MortonBlockOrder` 1 M points 27.6 → 12.5 ms and 10 M points 511 → 116 ms; `ApplyOrder` 10 M 245 → 115 ms. It does not scale past about 8 threads, because it is memory-bound: on 50 M points the project measured 80 Mpts/s at 8 threads and 75 at 32. The worker count is half the hardware threads divided by the number of active loaders, so the other half stays free for the render thread, the uploader and the driver (Post 07).

## Step 2: the surprise

Here is what you expect: Morton order puts neighbours together, so the GPU's caches and rasterizer should love it.

Here is what was measured: **Morton alone drew slower than file order.** The commit that introduced the ordering says so in its first line of justification ("Morton order alone draws slower with GL_POINTS"), and the header comment states it flatly: *"Morton alone is slower than file order; the block shuffle is what makes it fast."*

The fix is odd. After the sort, cut the array into full blocks of 128 points and **shuffle the blocks randomly** with a fixed seed. The trailing partial block stays last.

```cpp
constexpr std::size_t kOrderBlock = 128;     // points per shuffled block
// Gather order: points LSD-radix-sorted by (Morton code, index), then the full 128-point
// blocks shuffled with mt19937_64(seed). The trailing partial block stays last.
std::unique_ptr<uint32_t[]> morton_block_order(std::span<const glm::vec3> pos, glm::vec3 bbmin,
                                               glm::vec3 bbmax, uint64_t seed, ...);
```

Measured, at the time:

| change | result |
|---|---|
| one tile (L002, 10.3 M points), GL_POINTS draw | 5.2 → 1.4 ms with the shuffle |
| four tiles (78.3 M points), one buffer set per cloud | 58 → 32 ms (file order → Morton + block shuffle) |
| four tiles, with the pool from Post 03 | 25.5 ms (21-bit keys) → 24.0 ms (10-bit keys) |

Both rows of the 58 → 32 ms step used one buffer set per cloud, so that drop is the ordering alone. The further drop to 25.5 ms came with the pool, and the last one with the shorter keys.

### Why does a shuffle help? Honest answer: not proven here

The repository records *that* it helps and by how much. It does not contain a profiler capture that says *why*. What follows is a hypothesis, not a finding.

When a sorted cloud is drawn in order, consecutive warps work on points that land on the same few pixels, because the Morton curve is contiguous in space. In the fit-all view, 78 M points over 3.2 M pixels already means about 24 points per pixel. A long contiguous run of points that all hit the same screen tile makes the depth test and write units process the same location back to back, and the GPU's work distribution is unbalanced across its parallel units, so a hot tile queues while others idle. Shuffled blocks spread the concurrent work across the screen, while 128-point blocks stay internally coherent, which keeps each warp's loads together.

If you build this, *measure it on your own data and GPU*. The shuffle is cheap to try (a one-line change after the sort) and the direction of the effect may differ with your hardware. The `Order block` colour mode (below) and a GPU profiler will tell you more than this paragraph.

## Step 3: the block becomes the unit of everything

The 128 was not arbitrary, and it did not stay a GL_POINTS tweak. It is the compute raster's workgroup size and the unit of several later features:

- **The compute workgroup.** One workgroup of 128 threads handles exactly one block.
- **Culling (R3).** `block_bounds` computes a 24-byte axis-aligned box per block (15 MB at 78 M points). A workgroup tests the box's 8 corners against the frustum planes and exits if all 8 are outside the same plane. Because each block is spatially tight, the box is tight, and entire blocks are skipped. Measured close-up: raster time 1.10 → 0.344 ms (3.2x), fit-all +1.7 % (the boxes are bandwidth when nothing is culled), image diff 0 pixels. Post 15 covers it.
- **Mesh shader task culling.** The task shader culls the same blocks (Post 16).
- **LOD pages.** A streamed page is 4096 points, a multiple of 128, so the block is still the right unit.

The shuffle shapes what you can cull *by index*. A cloud's pool indices are not a spatial coordinate: a range of indices is a random set of blocks across the scene. That is fine for the block-level tests above, but it means you cannot cull by index range, and it is why the debug view below looks like confetti.

## Seeing it: two debug colour modes

Two colour modes exist purely to inspect the ordering (documented in `docs/pool-order-and-batches.md`):

- **Order block** colours each point by `hash(pool_index / 128)`: each patch on screen is one Morton-compact 128-point block, and the patch size does not change with the cloud size. A multiplicative hash is used because float precision runs out for block numbers in the tens of thousands (a golden-ratio hue step turns to noise).
- **Upload batch** colours by upload piece (Post 08). A 20 M point cloud is about 4 batches. Since blocks were shuffled over the whole cloud, every batch is a random quarter of all blocks, so each batch covers the whole scene and you see four colours interleaved at 128-point granularity. That picture is expected, not a failed upload.

Build the equivalent early: a colouring by block index is a few lines of shader and makes "is the order wrong?" answerable at a glance.

## Recipe

1. Quantize to **10 bits per axis**, Morton-interleave, **LSD radix sort** `(key, index)` pairs in 3 passes. Make it parallel, stable and cancellable.
2. Return a **gather order** and apply it to every attribute with one function.
3. **Shuffle full 128-point blocks** with a fixed seed, keep the tail last, and measure against the plain sort and against file order on *your* GPU.
4. Use that 128-point block as your workgroup, culling and paging unit.
5. Add an `Order block` debug colour on day one.

That finishes Part 1: data layout, one pool, camera-relative transforms, and a memory order the GPU likes. Part 2 gets the bytes onto the GPU, starting with the staging ring: [Post 06](../post06-persistent-mapped-staging-ring/).

## Where this lives in the code

`src/core/order.hpp` and `order.cpp` (`morton_codes`, `radix_sort_pairs`, `morton_block_order`, `apply_order`, `block_bounds`), `src/app/loading/load_job.cpp` (called on the loader thread), `benchmarks/bench.cpp` (`BM_MortonBlockOrder`, `BM_ApplyOrder`). Numbers: ROADMAP 2.2b, 2.2d and 3.1, BENCH.md "R3 - block culling", `docs/pool-order-and-batches.md`.
