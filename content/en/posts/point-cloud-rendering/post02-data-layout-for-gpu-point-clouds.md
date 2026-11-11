---
title: "Post 02: Data Layout for GPU Point Clouds"
date: 2026-10-14T11:00:00+03:00
weight: 3
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - memory-layout
---

[Post 01](../post01-why-gl-points-dies/) showed that the draw call is not where you win. The layout of the data decides what every later stage can do cheaply. This post covers the three decisions the editor made about a point before a single byte reaches OpenGL: structure of arrays, a double origin with float offsets, and one arena per cloud.

## What a point is

A point in this editor carries five things:

| attribute | type | bytes |
|---|---|---|
| position (relative to the cloud origin) | `float` x3 | 12 |
| colour | packed RGBA8 `uint32_t` | 4 |
| intensity | `uint16_t` | 2 |
| classification | `uint8_t` | 1 |
| flags (selected / hidden / deleted) | `uint8_t` | 1 |
| **total** | | **20** |

For the 78 320 210 Toronto-3D points that is about 1.57 GB, and the position array alone is 940 MB. Everything below is about spending those bytes well.

## Decision 1: structure of arrays

The obvious layout is an array of structs:

```cpp
struct Point { vec3 pos; uint32_t rgba; uint16_t intensity; uint8_t cls, flags; };  // 20 B, array of these
```

The cloud stores the opposite, one array per attribute:

```cpp
struct Cloud {
    Arena mem;                        // owns all arrays below
    glm::dvec3 origin{};              // double, absolute
    std::span<glm::vec3> pos;         // float, relative to origin
    std::span<uint32_t>  rgba;        // packed RGBA8 (R in the low byte)
    std::span<uint16_t>  intensity;
    std::span<uint8_t>   cls;
    std::span<uint8_t>   flags;       // kSelected = 1, kHidden = 2, kDeleted = 4
    glm::vec3 bbmin{}, bbmax{};       // relative to origin
    // ...
};
```

Why SoA, in order of how much it mattered:

**Each pass reads only what it needs.** The compute rasterizer (post 9) reads *only* `pos` for 78 M points. The resolve pass reads only the attribute of the active colour mode. This was measured: making the resolve fetch intensity, class and elevation inputs for every pixel, instead of only the active one, took RGB fit-all from 1.24 ms to 1.58 ms. With an interleaved struct, every position read drags 8 bytes of colour and metadata through the cache that the pass never uses.

**Attributes change at different rates.** Positions and colours are written once at load. Flags and class are edited constantly (select, hide, delete, reclassify), on the GPU, and undo is a delta over just those bytes (post 25). With SoA the edit passes touch a 1-byte-per-point buffer. With AoS they would read-modify-write a 20-byte record to flip one bit.

**It matches the GPU buffers one to one.** Each array becomes one GPU buffer (post 3). A format conversion on upload (AoS to SoA) would be a pass over 1.5 GB per load. With SoA the upload is a straight `memcpy` of each array into the staging ring (post 6).

**Cost.** SoA means five streams to keep in sync. Any code that moves points (compaction after deletes, merge, subsample, Morton reorder) must apply the same permutation to every array. The loader does the permutation in parallel: one sort key array, then one gather per attribute.

## Decision 2: a double origin, float offsets

Why not just store `double` positions? Because 24 bytes per point positions would be 1.9 GB for this dataset, and the GPU does float arithmetic. Why not just store `float` absolute coordinates? Because floats run out of precision fast. The spacing between adjacent representable floats (the ulp) at coordinate magnitude `m` is roughly `m / 2^23`:

| coordinate magnitude | float step |
|---|---|
| 1 000 m | ~0.06 mm |
| 100 000 m | ~8 mm |
| 5 000 000 m (UTM northing) | **0.5 m** |

That last row is a normal survey: UTM northings are in the millions of metres, and a float cannot represent a position finer than half a metre there. The cloud would collapse onto a grid, and nothing about the renderer could fix that afterwards.

So each cloud stores:

- one `dvec3 origin` in absolute coordinates, and
- one `vec3` per point, *relative to that origin*.

The origin is chosen at load time so offsets stay small: `floor` of the file's first vertex for PLY, `floor` of the header minimum for LAS. A tile that spans a few hundred metres then has offsets of a few hundred metres at most, which a float holds to well under a millimetre.

The absolute position is recovered only when needed (picking, measuring, saving), in double, as `origin + pos`. The save path writes PLY as double `x y z` from exactly that sum.

Two clouds have two origins, so how do they draw together? The model matrix is built in double from the difference between the cloud's origin and the camera, then cast to float. The GPU only ever sees numbers near the camera. That is the subject of [Post 04](../post04-camera-relative-rendering-and-reverse-z/); here it is enough to say that the layout is what makes it possible.

The project checks the precision claim with a test: a PLY (double) and a LAS (0.001 m scale) with points about 10 m and about 1 km apart at UTM magnitude must give an `origin + pos` distance within 1e-4 m (PLY) and 1.1e-3 m (LAS).

## Decision 3: one arena per cloud

The five arrays are not five `std::vector`s. They are carved from one allocation:

```cpp
struct Arena {                         // fixed capacity, bump allocator, never grows
    static constexpr std::size_t kAlign = 64;
    explicit Arena(std::size_t capacity)
        : base_(static_cast<std::byte*>(::operator new(capacity, std::align_val_t{kAlign}))), cap_(capacity) {}

    template <class T> std::span<T> alloc(std::size_t n);   // uninitialized, 64-byte aligned
};
```

`cloud_arena_bytes(n)` computes the exact size for `n` points (the array sizes plus worst-case alignment padding per array), so the loader allocates once, knowing the point count from the file header.

What this buys:

- **No zero-fill and no reallocation.** Allocation returns uninitialized storage (the types are required to be trivially constructible). A `vector<vec3>` of 78 M elements would write zeros first, then you overwrite them, and growing it would copy 940 MB. A fixed-capacity bump allocator does neither.
- **64-byte alignment per array.** Every array starts on a cache line, so parallel loader threads writing neighbouring chunks of the same array don't share lines at the start, and SIMD loads are aligned.
- **A single free.** Dropping a cloud is one `operator delete`.
- **Mappable sources.** For uncompressed LAS the loader memory-maps the file and parallelizes over the record stride, writing straight into the arena.

Caveats worth knowing: the arena never grows, so the point count must be known before allocation (true for PLY, LAS and PCD headers), and moving a `Cloud` moves the arena pointer but not the memory. Background readers such as a save job therefore take a `cloud_view`, the same spans without the arena, and the owner has to outlive the view.

## Where the truth lives

One more layout fact surprises people: for a regular cloud, **the CPU arrays are not the truth after load.** `flags` and `cls` are edited in place on the GPU, and the CPU copies go stale until a save or merge reads them back. `pos`, `rgba` and `intensity` are never edited, so the CPU and GPU copies stay equal. This split is only possible because the attributes are separate arrays; post 24 builds on it.

## Recap

- Store **SoA**: each pass reads only the bytes it needs, and the arrays map straight onto GPU buffers.
- Store a **double origin plus float offsets**: floats alone give 0.5 m steps at UTM magnitudes.
- Allocate each cloud from **one aligned arena**: no zero-fill, no regrowth, one free.
- Keep **editable attributes separate** from immutable ones, because the GPU will edit them.

Next: how these five arrays become GPU buffers without one allocation per file, in [Post 03: One big pool, not one buffer per cloud](../post03-one-big-pool/).

## Where this lives in the code

`src/core/cloud.hpp` (Cloud, Flag), `src/core/arena.hpp` (Arena), `src/render_model/render_model.hpp` (`Attr`, per-attribute byte sizes). Numbers come from `docs/ROADMAP.md` items 1.2, 2.0, 2.4, 3.3 and 5.5 and from the "Cloud data model" section of `docs/rendering-engine.md`.
