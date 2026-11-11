---
title: "Post 06: A Persistent-Mapped Staging Ring for Point Uploads"
date: 2026-10-14T15:00:00+03:00
weight: 7
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - upload
---

Part 1 ended with a cloud in memory, ordered, in five attribute arrays. Now it has to reach the pool from [Post 03](../post03-one-big-pool/) without freezing the UI. About 1.57 GB of data for 78 M points has to cross, and the render thread owns the GL context. This post covers the upload path: a ring of three 64 MB slots, persistently mapped, filled by one thread and drained by another.

## Why not `glNamedBufferSubData`?

The simple call copies from pageable memory. The driver has to pin or stage the pages itself, and the call blocks the thread that made it. The project measured it:

| path | throughput |
|---|---|
| `glNamedBufferSubData` from pageable memory | 1.1-1.7 GB/s |
| persistent-mapped staging + `glCopyNamedBufferSubData` | 9.5-10 GB/s |

On four Toronto tiles (78.3 M points, 1.2 GB of pos + rgba) the pieces measured separately were: `memcpy` into staging 14.0 GB/s, the GPU-side copies 19.2 GB/s, and the two together 8.1 GB/s. Everything was uploaded 4.9 s after start, and after the loader work in Post 07, 2.3 s.

The design goal is not only speed but **where the work happens**. The render thread should issue a handful of cheap GPU-side copies per frame, never a big memcpy or a driver stall.

## The ring

One buffer, three slots of 64 MB, mapped once and never unmapped:

```cpp
constexpr int kSlots = 3;
constexpr std::size_t kSlotBytes = 64u << 20;

void staging_init(Staging& s)
{
    constexpr auto flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glCreateBuffers(1, &s.buf);
    // CLIENT_STORAGE_BIT: without it, NVIDIA on Windows pins all 192 MB inside the map call (~5 s at startup).
    glNamedBufferStorage(s.buf, kSlots * kSlotBytes, nullptr, flags | GL_CLIENT_STORAGE_BIT);
    s.ptr = static_cast<std::byte*>(glMapNamedBufferRange(s.buf, 0, kSlots * kSlotBytes, flags));
    s.uploader = std::jthread(uploader_main, std::ref(s));
}
```

- `GL_MAP_PERSISTENT_BIT` lets the CPU keep the mapping while the GPU reads the buffer. `GL_MAP_COHERENT_BIT` removes the need for manual flushes.
- `GL_CLIENT_STORAGE_BIT` is a hint that the storage should live in CPU-visible memory. The comment in the code records the reason: on NVIDIA/Windows, mapping without it pinned the whole 192 MB inside the map call and added about 5 s to startup. (It is a hint, so check that your driver treats it the same way.)

## Three owners, four states

A slot is owned by exactly one party at a time:

```cpp
enum class SlotState : uint8_t { Free, Filling, Ready, InFlight };
```

| state | who touches it | what is happening |
|---|---|---|
| `Free` | uploader thread may take it | no copy referencing it is pending |
| `Filling` | uploader thread | `memcpy` / decode into the mapped memory |
| `Ready` | render thread | filled, copy not issued yet |
| `InFlight` | GPU | `glCopyNamedBufferSubData` issued; fence pending |

The transitions are the whole synchronization protocol:

1. **Uploader thread** (`uploader_main`): waits on a condition variable until a slot is `Free` and tasks are queued. It packs as many queued tasks as fit into 64 MB, in order, marks the slot `Filling`, fills it, and stores `Ready` with release semantics.
2. **Render thread** (`staging_pump`, once per frame): first retires `InFlight` slots whose fence has signalled, then, for each `Ready` slot, issues one `glCopyNamedBufferSubData` per task from the ring into the destination pool buffer, adds one fence per slot (`glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE)`), and marks it `InFlight`.
3. **A slot returns to `Free`** only after its fence signalled. That rule is what makes the mapped memory safe to overwrite: the GPU is done reading it.

```cpp
for (StagingSlot& sl : s.slots) {
    if (sl.state.load() != SlotState::InFlight) continue;
    const GLenum r = glClientWaitSync(sl.fence, SyncObjectMask{}, 0);   // timeout 0: poll, never block
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) continue;
    glDeleteSync(sl.fence);
    // ... return the tasks to the caller, then:
    { std::lock_guard lk(s.m); sl.state = SlotState::Free; }      // under the lock, or the uploader can miss the wakeup
    s.cv.notify_one();
}
```

That comment about the lock is the classic lost-wakeup bug, fixed in the code and worth copying: change the predicate's state under the same mutex the waiter uses.

Why three slots? My reading, not a measurement from the repo: three lets one slot be filled, one wait for the render thread, and one be in flight, so neither side idles on the other. It is a small constant (`kSlots`) and cheap to change; measure on your hardware.

## What a task is

The ring moves **tasks**, not clouds:

```cpp
struct UploadTask {
    uint64_t cloud_id;
    int attr;                   // pool buffer (Attr)
    const std::byte* src;       // into the cloud's arena; must outlive the task
    std::size_t size;           // <= kSlotBytes; bytes written to the pool (decoded)
    std::size_t dst;            // byte offset in the pool buffer of `attr`
    uint8_t codec;              // copy, or a .pco v2 decode
    glm::dvec3 base; double scale;
};
```

A loaded cloud is cut into pieces of `kUploadChunkPoints = 64 MB / 12 B = 5 592 405` points, and each piece produces one task per attribute (pos, rgba, intensity, class) at its own offset in the pool. Because the source is the cloud's own arena (Post 02), "stage" is one `memcpy` per task, and `src` just has to outlive the task. The `codec` field lets the uploader thread *decode* instead of copy, which is how the quantized `.pco` v2 nodes of Post 19 are expanded off the render thread.

Two consequences worth noting:

- **The cloud draws progressively.** Each piece covers all attributes of the same point range, and the scene counts a point as drawable only when all its attributes have landed (`uploaded()` takes the minimum over the per-attribute byte counts). Big files appear in 5.6 M-point steps, not all at once.
- **Mixed task sizes need packing.** The first version put one task per slot. That was fine for 64 MB pieces but disastrous for LOD pages of 8-48 KB: a slot would carry one page and the ring's throughput collapsed. The uploader now packs queued tasks in order until the next one would not fit, and the render thread issues one copy per task. Even for big clouds the code carries a note that the intensity and class tasks fill only a sixth and a twelfth of a slot, accepted as a known inefficiency.

The uploader fills a slot with a small number of worker threads (`parallel_for`, default 4), so decode and page faults on a cold memory-mapped file overlap (Post 20).

## Rules that keep it correct

- **GL calls stay on the render thread.** The uploader thread never touches GL; it writes into already-mapped memory. Parsing and decoding happen outside the context.
- **The source must outlive the copy.** `src` points into the cloud's arena. A cloud cannot be freed while a task naming it is queued or in flight. `staging_pump` returns finished tasks to the caller for exactly this accounting, and `release()` stops and joins the uploader *before* the clouds it reads from are freed.
- **Compaction must drain first.** `pool_realloc` moves ranges (Post 03). Tasks target pool byte offsets, so `staging_drain` pumps until every queued task has retired before a compaction runs. It waits on one fence at a time with a 1 ms timeout instead of spinning.
- **Flags are never staged.** The flag buffer is cleared on the GPU at ingest with `glClearNamedBufferSubData`, so zero-filled flags cost no ring bandwidth.

## Before you copy it

1. Start with the numbers: measure `SubData` vs ring on *your* driver. The 6-9x gap is one machine's result.
2. Use a handful of fixed slots with a state machine and fences, not a ring of tiny regions. 64 MB slots keep the fence count low.
3. Poll fences with a zero timeout. Blocking in `glClientWaitSync` on the render thread reintroduces the stall you built the ring to avoid.
4. Give each task an explicit lifetime contract (`src` must outlive it) and test the shutdown order.
5. Make the first-write placement right: the ring and the pool buffers are fresh buffers, and the NVIDIA trap from Post 03 applies to the destinations.

Next: the producer side of this ring, loading 100 M points without freezing, in [Post 07](../post07-loading-100m-points-without-freezing/).

## Where this lives in the code

`src/render/upload.hpp` and `upload.cpp` (the ring), `src/render_model/upload_task.hpp` (`UploadTask`, `kSlotBytes`, `kUploadChunkPoints`, codecs), `src/app/loading/ingest.cpp` (task creation). Commits: b067f0a (first ring), 16e6e6b (packing small uploads), 4bd7157 (client-storage staging). Numbers: ROADMAP 2.2.
