---
title: "Post 19: Quantized Nodes, 12 Bytes per Point on the Source's Grid"
date: 2026-10-15T13:00:00+03:00
weight: 20
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - compression
---

The first `.pco` format ([Post 18](../post18-out-of-core-octree/)) stored each node in the same layout as the GPU pool: float xyz relative to one origin, packed RGBA, intensity, class. That is 19 bytes per point, and for `toronto16` it came to 23.8 GB. Disk size matters twice here: it is what a user stores, and it is what a cold read has to pull in ([Post 20](../post20-streaming-from-a-cold-disk/)). This post covers version 2: 12 bytes per point, with no loss against the source file.

## What is wasted in v1

- **Float positions hide a real limit.** Floats relative to one octree origin have a step that grows with the cube: about 0.25 mm over a 4 km cube. For a survey that stores a 1 mm grid, that is fine for display but it silently *re-rounds* data when you save.
- **Colour takes four bytes** (RGBA) even for sources that have no alpha; v2 stores three and keeps a header flag for sources that do.
- **Positions use 12 bytes** where the data needs less, since inner and leaf nodes span very different extents.

## The format

Positions are **integers on a global grid**, stored relative to the node:

```
per node:   OctNodeHead { int32 base[3]; wide }            16 bytes
            x y z       u16 offsets from base   (u32 when the node spans >= 65 536 steps)
            intensity   u16
            class       u8
            colour      RGB 3 bytes (RGBA if the header has kOctAlpha)
```

For most nodes that is `6 + 2 + 1 + 3 = 12` bytes per point. The width choice is per node: a leaf spanning a few metres uses 16-bit offsets, a big inner node spans more than 65 536 grid steps and takes 32-bit offsets. An earlier idea, "16 bits over the node's extent", was rejected in the design notes because it would be lossy on large inner nodes, and inner-node points are real points that get exported.

**Which grid?** The inputs' own: if all input files share a LAS grid (same scale, offsets a whole number of steps apart), the octree uses that grid, with the finest scale, so *every source integer is kept exactly*. Otherwise (PLY, PCD, mixed) it uses 0.0001 m from `floor(min)`, configurable. The grid origin is moved by whole steps to keep the integers inside `int32`, and coarsened only if the data spans more than 2 × 10⁹ steps. The converter carries grid integers through its temp file (the same 20 bytes per point), so the tree is built from the quantized positions and nothing is re-rounded later.

## Decode on the uploader thread

The renderer was not changed. The staging ring from [Post 06](../post06-persistent-mapped-staging-ring/) has an `UploadTask::codec`, and v2 nodes are decoded while they are copied into the slot:

```cpp
enum UploadCodec : uint8_t { kUploadCopy, kUploadPos16, kUploadPos32, kUploadRgb };

case kUploadPos16:
case kUploadPos32:
    oct_decode_pos(t.src, t.codec == kUploadPos32, t.base, t.scale, t.size / sizeof(glm::vec3), dst);
    break;
case kUploadRgb:
    oct_decode_rgb(t.src, false, t.size / 4, dst);   // RGB -> RGBA with alpha 255
    break;
```

Positions are decoded in double (`base + q * scale`) and stored as float offsets in the pool's layout, so everything after upload ([Post 09](../post09-software-rasterizing-points-with-64-bit-atomics/) and onwards) is identical for v1 and v2. The design notes list a second step, keeping quantized data in VRAM and decoding in the raster and resolve shaders, and mark it **not needed**: the LOD cache uses 0.7 to 2.4 GB of a 24 GB card. Do the simple decode first; do the shader decode only if VRAM becomes the limit.

## What it bought

`toronto16` (input LAS 32.6 GB, 64 threads), back to back:

| layout | file | bytes/point | conversion |
|---|---|---|---|
| v1 (float) | 23.82 GB | 19.01 | 155.3 s |
| **v2** | **15.14 GB** | **12.08** | **144.6 / 143.8 s** |

37 % smaller, with conversion no slower (slightly faster; most of the gain is in the index pass, 86 s against 75-77 s). Streaming with the then single uploader thread (decoded MB/s, cold from a fresh unbuffered copy):

| view | v1 cold | v2 cold | v1 warm | v2 warm |
|---|---|---|---|---|
| fit all | 147 | 275 | 1198 | 1184 |
| aerial 300 m | 219 | 367 | 2039 | 1537 |
| street 60 m | 240 | 381 | 2099 | 1562 |
| street 15 m | 232 | 383 | 1942 | 1513 |

Cold got 1.6 to 1.9x faster (37 % fewer bytes to fault in, and fewer pages per node). Warm got about 25 % **slower**, because the one uploader thread now decodes as well as copies. The measurement is recorded as it fell, and the next fix ([Post 20](../post20-streaming-from-a-cold-disk/)) is what moved both numbers: four threads per slot.

## Exactness, tested and bounded

The promise is "every LAS integer survives", and it is checked, not assumed:

- a test builds a v2 file from a LAS and checks the integers in the file, and again after a save (`check_octree_v2`: bit-exact on a 3 x 2 km set), including v1 decoded equal to v2 and the alpha flag;
- **saving an unmoved v2 cloud writes LAS/LAZ on the stored grid**: the source's integers, scale and offset come back out;
- v1 files still open and stream; old caches are reused while their manifest matches, and deleting one rebuilds it as v2.

One honest caveat is in the notes. Exactness needs the *reader's* float positions to sit within half a grid step of the true integer. The converter measures this (`OctStats::grid_dev`, must be below 0.5) and reports it. The practical limit: a **1 mm grid is exact for extents up to about 8 km** from the file's `floor(min)`, and a **0.1 mm grid up to about 800 m**. Beyond that the float intermediate is too coarse to recover the integer. A wider file would need the reader to keep positions in double; I did not find such a path in the repository.

## Checklist

1. **Quantize on the source's own grid** when it has one; you can then promise losslessness.
2. Choose **offset width per node**, not globally.
3. Keep **decoding off the render thread and out of the renderer**, using a codec field on the upload task.
4. Delay in-VRAM compression until a measurement says VRAM is the problem.
5. Keep reading the old version, and keep a **bit-exactness test** for the new one.
6. Write down the **range of validity** of the exactness claim.

Next: a 12-byte format reads 37 % fewer bytes, but the first cold run was still slow. Why, and how it got 12x faster: [Post 20](../post20-streaming-from-a-cold-disk/).

## Where this lives in the code

`src/io/octree.hpp` (v2 layout comments, `OctHeader`, `oct_decode_pos`, `oct_decode_rgb`), `src/render_model/upload_task.hpp` (codecs, `upload_fill`), tests `check_octree_v2` / `check_octree`. Numbers: FIXES F2, BENCH.md "F2 - .pco v2". Commit 6c5d9d9.
