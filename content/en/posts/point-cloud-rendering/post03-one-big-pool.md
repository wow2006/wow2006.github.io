---
title: "Post 03: One Big Pool, Not One Buffer per Cloud"
date: 2026-10-14T12:00:00+03:00
weight: 4
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - ssbo
---

[Post 02](../post02-data-layout-for-gpu-point-clouds/) ended with five attribute arrays per cloud. The obvious next step is five `glNamedBufferStorage` calls per cloud and one draw per cloud. This post explains why the editor does something else: one pool of five buffers for *everything*, addressed by a global point index, bound to shaders in chunks that respect the SSBO size limit.

## The pool

```cpp
struct GpuPool {
    gl::GLuint vao = 0, buf[kAttrs]{};   // indexed by Attr: kPos, kRgba, kIntensity, kCls, kFlags
    gl::GLuint fresh_vao = 0;            // same formats as vao, for first_draw
    gl::GLuint zero = 0;                 // 16 zero bytes
    std::size_t capacity = 0, used = 0;  // points
};
```

Five buffers, `capacity` points long each. A cloud does not own a buffer. It owns a **contiguous range** `[first, first + count)` of point indices that is the same in all five buffers. Loading a second file appends a range. Removing a cloud frees a range (and the pool can compact, below).

Why a pool and not one buffer set per cloud:

**The pixel's answer is one number.** The compute rasterizer (post 9) packs `depth << 32 | point index` into a 64-bit value per pixel. With a pool, that index is a *global* index, identical for every cloud. The resolve pass turns it into a colour by reading `rgba[index]`, and finds which cloud it belongs to by a binary search over a small sorted range table (`CloudRange {first, end, zoff, tint}`, 32 bytes each). With per-cloud buffers the pixel would need a (buffer, index) pair and the resolve would need to bind every cloud's buffers.

**Everything downstream uses the same index.** Picking returns a pool index. Edit passes write `flags[index]`. Undo deltas are XOR bitmasks over pool positions. The streamed LOD pages (post 21) are ranges in the same pool, so the same shaders run on them.

**Shader binding counts are finite.** The resolve already uses 14 SSBO blocks in the fragment stage, and the hardware limit is not large. A scene of 30 files cannot bind 150 buffers.

**One allocation for the driver.** Two or three big buffers are cheaper to manage and to keep resident than hundreds of small ones.

The cost: ranges must be allocated and compacted by us. That is the next section.

## Sizing: whole words only

The compute shaders read `intensity`, `cls` and `flags` as `uint[]`, four or two elements per word. So each buffer is rounded up to a multiple of 4 bytes:

```cpp
for (int a = 0; a < kAttrs; ++a) // whole 4-byte words: the resolve reads intensity and cls as uint[]
    glNamedBufferStorage(buf[a], (capacity * kAttrBytes[a] + 3) & ~size_t(3), nullptr, BufferStorageMask{});
```

and a byte at point `i` is extracted with a shift:

```glsl
(flags[i >> 2] >> ((i & 3u) * 8u)) & 0xffu
```

GLSL has no 8-bit SSBO element in core without extensions, so this is the portable way, and it keeps the buffers tight: flags and class cost 1 byte per point, not 4.

## Growth and compaction: always a fresh buffer

`pool_realloc(pool, new_capacity, moves, program)` takes a list of `{src, dst, count}` moves and does:

1. create five new buffers of the new capacity,
2. `glCopyNamedBufferSubData` each move from old to new, GPU-side,
3. rebind the VAO, delete the old buffers.

Growth is one move `{0, 0, used}`. Compaction after a delete is the surviving ranges packed to the front. It has to be a *fresh* buffer: `glCopyBufferSubData` forbids overlapping source and destination ranges inside one buffer, so you cannot slide ranges within the buffer.

Honest cost: the resize is synchronous work. It is documented as a known exception to the project's no-freeze rule, measured at 50-480 ms for 10-290 M points. A header-sized pool (the loader knows the point count up front) avoids most resizes; the rest is the price of the simple design.

### An NVIDIA placement trap

On driver 610.57, a buffer's memory placement is decided at its **first draw**. If the buffer is written first (a copy, a clear, a shader write), it is created in video memory, moved to host memory at the first draw, and stays slow until it migrates back. The log message is debug id 131186, the stall was about 1 second per 2 GB.

The workaround is a one-point discarded draw before any write:

```cpp
glEnable(GL_RASTERIZER_DISCARD);
glUseProgram(program);          // the point program reads every attribute
glBindVertexArray(p.vao);
glDrawArrays(GL_POINTS, 0, 1);  // placement decided here, in VRAM
glDisable(GL_RASTERIZER_DISCARD);
```

`pool_realloc` does this on every fresh set of buffers, and `first_draw()` does it for other new buffers (undo deltas had the same problem). A second trap lives in the same code: the fresh VAO must use exactly the same vertex formats as the real one, or the driver recompiles the shader (debug id 131218).

## The SSBO size limit and chunked bindings

A pool of 235 M points has a 2.8 GB position buffer. Shaders cannot bind that as one SSBO: `GL_MAX_SHADER_STORAGE_BLOCK_SIZE` is 2 GiB on the test machine. At 12 bytes per point that is **178 M points per binding**. The buffer itself can be bigger than the limit. It is the *binding* that is limited, so you bind a window over it.

`glBindBufferRange(target, index, buffer, offset, size)` binds a window, but `offset` must be a multiple of `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`, which is not a multiple of 12. A point range can start anywhere. The helper:

```cpp
struct SsboChunk { size_t offset, size, skip, count; };   // bytes, bytes, words, points
constexpr SsboChunk ssbo_chunk(size_t first, size_t count, size_t stride, size_t align, size_t max_bytes)
{
    const size_t start = first * stride, offset = start & ~(align - 1), lead = start - offset;
    const size_t n = std::min(count, (max_bytes - lead) / stride);
    return {offset, lead + n * stride, lead / 4, n};
}
```

A worked example, with a hypothetical alignment of 256 bytes: a cloud starts at point 1000, `start = 12000`, the aligned offset is `11776`, `lead = 224` bytes, `skip = 56` words. The shader then reads point `i` of the cloud at float index `skip + 3 * i`. The window is bound at 11776, and the first point is 224 bytes into it. If the cloud is bigger than one window, the function returns how many points fit (`count`), and the host loops:

```cpp
while (count) {
    const SsboChunk c = ssbo_chunk(first, count, sizeof(glm::vec3), ssbo_align, max_block);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, pool.buf[kPos], c.offset, c.size);
    glProgramUniform1ui(program, 1, c.count);   // points in this dispatch
    glProgramUniform1ui(program, 2, c.skip);    // floats before the first point
    glProgramUniform1ui(program, 3, first);     // global index of the first point (for the 64-bit key)
    glDispatchCompute((c.count + 127) / 128, 1, 1);
    first += c.count; count -= c.count;
}
```

Because `first` is passed as a uniform, the 64-bit key written to the framebuffer still contains the *global* pool index no matter which chunk produced it.

### `float pos[]`, not `vec3 pos[]`

```glsl
layout(std430, binding = 0) restrict readonly buffer Pos { float pos[]; }; // float, not vec3: vec3[] pads to 16 B
...
const uint j = skip + 3 * i;
vec4 c = mvp * vec4(pos[j], pos[j + 1], pos[j + 2], 1.0);
```

In `std430`, an array of `vec3` has a stride of 16 bytes, not 12. Declaring `vec3 pos[]` would read garbage from a tightly packed buffer (or force you to store 33 % padding). A `float` array of 3 N elements has no padding and a 12-byte logical stride that you index yourself. This is the single most common bug when moving point data into an SSBO.

### Chunks for the resolve

The resolve (a fragment shader) reads `rgba` (4 B/point) for the whole pool, and in elevation mode `pos` too. Its bindings are fixed at draw time and there are only so many, so it uses up to **4 chunks** of equal size, `points = (max_block / 12) & ~(align - 1)` points each. Point `i` is in chunk `i / points`, at `i % points` inside it. With a 2 GiB limit that is room for about 715 M points; a larger pool falls back to `GL_POINTS`, and the Stats panel says why (`GL_POINTS (pool over 4 chunks of max block)`). The helper is `resolve_chunks()` in `render_model.hpp`, and a test forces tiny limits (`PCE_MAX_BLOCK=...`) so the chunk loop runs on small data. In the measured 235 M-point scene (224 M pool), elevation colouring ran on the compute path because of this chunking.

## What to copy

1. **One pool, global indices.** Anything that has to refer to a point (framebuffer, pick, edit, undo) gets simpler.
2. **Keep the GL-free layout code testable.** `ssbo_chunk`, `resolve_chunks` and `choose_renderer` are `constexpr` functions in a header with no GL types, so Catch2 tests exercise the boundary cases (empty ranges, exact-fit, offset leads) without a context.
3. **Round to words, use `float[]` for positions.**
4. **Grow by fresh buffers.** And draw one discarded point before writing them, if you ship on NVIDIA.
5. **Bind windows, not buffers.** `glBindBufferRange` plus a `skip` uniform gets past the SSBO size limit.

Next: the numbers in the pool are only useful if neighbouring points are neighbours in memory. [Post 04](../post04-camera-relative-rendering-and-reverse-z/) covers how cameras and depth work so far-away data stays precise; [Post 05](../post05-morton-order/) then reorders the pool.

## Where this lives in the code

`src/render/render.cpp` (`pool_realloc`, `first_draw`, the raster loop), `src/render_model/render_model.hpp` (`Attr`, `ssbo_chunk`, `ssbo_chunk_blocks`, `resolve_chunks`, `CloudRange`, `range_index`), `src/app/gl/gl_info.cpp` (the limit query and `PCE_MAX_BLOCK`). Context: `docs/rendering-engine.md` "Pool and chunking", ROADMAP 3.2 and 3.3, FIXES F4.1 and F4.4.
