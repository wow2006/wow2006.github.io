---
title: "Post 07: Loading 100M Points Without Freezing the UI"
date: 2026-10-14T16:00:00+03:00
weight: 8
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
  - threading
---

[Post 06](../post06-persistent-mapped-staging-ring/) built the consumer side of the upload path. This post is the producer: how a PLY, LAS, LAZ or PCD file becomes an ordered `Cloud` on a worker thread, how that thread hands it to the render thread, and how the work is spread so that reading, sorting and uploading four files at once does not freeze the window.

## The shape of the pipeline

```
file  -> [loader thread, one per file]  load_cloud -> order_cloud (Morton + gather + block boxes)
      -> parsed queue (mutex)
      -> [render thread] ingest(): VRAM check, pool range, upload tasks, GPU flag clear
      -> staging ring (Post 06) -> pool
```

Only the last two steps touch OpenGL. Everything before the queue runs on a `std::jthread` per file, and the queue is the only shared state.

```cpp
e.job->expected = point_count(p).value_or(0);   // header only, well under 1 ms from the page cache
...
e.job->thread = std::jthread([job, id, pq](std::stop_token st) {
    tracy::SetThreadName(("load " + job->path.filename().string()).c_str());
    auto cs = load_cloud(job->path, &job->progress, st);
    for (Cloud& c : *cs) boxes.push_back(order_cloud(c, st));   // Morton sort + apply + per-block boxes
    std::lock_guard lk(pq->m);
    pq->items.push_back({id, std::move(cs), std::move(boxes)});
});
```

Three conventions come from the project rules and show up in every job:

- **Errors are values.** `load_cloud` returns `std::expected<std::vector<Cloud>, std::string>`. A bad file puts its message on the row in the UI instead of throwing across a thread.
- **Progress and cancel are built in.** Each loader gets a `std::atomic<float>` progress that the UI reads, and a `stop_token` that the readers poll (every 65 536 points inside the LAZ decode, per radix pass in the sort). A cancelled load returns an error and frees its arena.
- **Name your threads.** Every worker calls `tracy::SetThreadName`, so the Tracy timeline reads as "load L003.ply" instead of anonymous threads.

## Step 1: read the header first, size the pool once

`point_count(path)` reads only the header. With it, the app knows how many points are coming *before any loader starts*, and `presize()` grows the pool once for everything pending:

```cpp
void App::presize()
{
    std::size_t pending = 0;
    for (const SceneCloud& o : scene) if (o.job) pending += o.job->expected;
    if (pool.used + pending > pool.capacity && fits(pool.used + pending))
        pool_realloc(pool, pool.used + pending, std::array{PoolMove{0, 0, pool.used}}, *program);
}
```

For four files at startup this turns three pool growths into one (27-34 ms) and runs it *before* the loaders start. The same growth done later next to 60 busy worker threads took 70-115 ms. `fits()` checks the pool against the VRAM budget (`GL_NVX_gpu_memory_info`, minus a margin), and when the total does not fit, ingest falls back to per-cloud growth and refuses a cloud that cannot fit, with a message ("not enough VRAM: pool needs N MB, M MB free").

## Step 2: do not start loaders with the first frame

When the loaders started together with the first frame, the first swap took 150-200 ms. Loaders now start after the first frame is presented. In the same spirit the window is created hidden and shown after the first frame (commit 4bd7157), so users never see a blank window while GL initializes.

## Step 3: how many threads?

Loading is memory-bound. The PLY vertex loop stops scaling at about 16 threads, and with 62 of 64 threads busy the render thread's frames hit 54-75 ms, because the main thread, uploader, GL driver and compositor had no cores left. So the rule is:

```cpp
// Loader threads currently running; each loader's parallel_for gets an equal share of the worker budget.
inline std::atomic<int> g_active_loaders{0};

inline std::size_t parallel_workers(std::size_t n, std::size_t min_chunk, std::size_t max_threads = 0)
{
    if (max_threads == 0)
        max_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency() / 2
                                                   / std::size_t(std::max(1, g_active_loaders.load())));
    return std::clamp<std::size_t>(n / min_chunk, 1, max_threads);
}
```

**Half the hardware threads, divided among the active loaders.** `g_active_loaders` is incremented at spawn time, not inside the thread, so loaders started together all see each other. `parallel_for` is a plain fork-join over `std::jthread` (no TBB, no `std::execution::par`, which would drag TBB in under libstdc++), with worker 0 on the calling thread and chunk `w` being `[n*w/T, n*(w+1)/T)`. Same arguments give the same chunks, so results are independent of the thread count, and tests compare `max_threads = 1` against many.

The code states its own ceiling: the share is sampled when a call starts, so a loader that finishes does not hand its threads to the others mid-call. A shared work-stealing pool would fix that "if it ever shows up in a profile". Each call also spawns its own `jthread`s, about 30 µs each. Morton and gather stop scaling at about 8 threads (80 Mpts/s at 8 against 75 at 32 on 50 M points), and on 1 M points 32 threads are slower than 8 (13.0 against 7.6 ms), where the spawn cost is a plausible part of the bill.

## Step 4: the readers

`load_cloud` dispatches by extension (`.ply .las .laz .pcd .e57 .usd*`).

**PLY and uncompressed LAS are memory-mapped, and conversion runs in parallel over ranges of the mapped file.** The reader computes the file's origin (`floor` of the first vertex for PLY, `floor` of the header minimum for LAS, Post 02) and converts each record to float offsets, packed RGBA and intensity straight into the cloud's arena. Prefaulting the mapping (`MAP_POPULATE`, `MADV_POPULATE_READ`) cost +20 ms serial and gave no gain; `MADV_WILLNEED` changed nothing, so none are used. Measured (median of 3, serial to parallel): `LoadPly` on a 10.3 M-point tile 316 → 68 ms, a synthetic 10 M-point file 338 → 74 ms.

**LAZ was 47x slower than LAS** (2.36 M points/s) because laz-perf's reader is serial. The fix reads the file's **chunk table** (LAZ stores independent compressed chunks, usually 50 000 points each) and decodes one chunk per task with its own decompressor, each writing straight into its slice of the output:

| threads | 10 M points, 200 chunks | speed-up |
|---|---|---|
| 1 (serial) | 4238 ms | 1.0x |
| 8 | 635 ms | 6.7x |
| 16 | 350 ms | 12.1x |
| 32 (default) | 249 ms | 17.0x |
| 64 | 246 ms | 17.2x |

Past 32 threads the rows stop moving (SMT siblings and memory bandwidth on the 260 MB record buffer). For comparison the same data as uncompressed LAS loads in 78 ms.

The safety work is the interesting part, because compressed files are untrusted input. The chunk table is validated against the file: offsets inside the file, version 0, the chunk count equal to `ceil(n / chunk_size)`, sizes that do not run into the table, counts that sum to the header's `n`, no zero-size chunks. **Anything odd falls back to the serial reader.** Each chunk decodes through a bounds-checked reader of its own byte range, so a corrupt chunk throws "LAZ decode failed" instead of reading out of bounds. Tests assert that the parallel output is byte-identical to the serial output, over 7 chunks with a partial last, a single chunk, variable-size chunks, LAS 1.2 and 1.4, and thread counts 0, 2, 5 and 64. They were mutation-tested, which means breaking the code on purpose (wrong output offset, wrong last-chunk count, dropped bounds check) to confirm that a test fails.

**Files larger than RAM** use `read_batches` instead: the same conversion, but delivered as views of at most N points valid only during the callback, which is the input side of the out-of-core converter in Post 18.

## Step 5: ingest on the render thread

The render thread takes finished clouds off the queue once per frame in `ingest()`. For each cloud it:

1. checks the VRAM budget and grows the pool if needed (`max(2 * capacity, used + n)`),
2. assigns `[base, base + n)` in the pool,
3. computes intensity percentiles from every 61st point (min/max made Toronto's median 4 % grey; the 1st to 99th percentile fixes the contrast),
4. queues the upload tasks in `kUploadChunkPoints` pieces (Post 06),
5. creates the per-block box buffer for culling (Post 15) with a first-draw guard,
6. clears the flag range on the GPU,
7. fits the camera if auto-fit is on.

The expensive CPU work (parse, sort, gather, box computation) is already done on the loader thread, so this stays short.

## Result

Four Toronto tiles (78.3 M points), vsync on:

| | before | after |
|---|---|---|
| all points uploaded | 4.9 s | 2.3 s |
| `load_ply` on the largest tile | 1.25 s | 0.56 s |
| Morton + gather (that tile) | 3.3 s | 1.45 s |
| pool growths | 3 | 1 |
| worst CPU frame while loading | 70 ms | 34-49 ms |

## Checklist

1. Read the header first; size the GPU pool once for everything pending.
2. Do all parsing, sorting and decoding off the GL thread; the queue is the only shared state.
3. Give every job a progress atomic and a stop token, return errors as values.
4. Budget threads: half the machine, divided among concurrent loaders. Measure where yours stops scaling.
5. If your format has independent compressed chunks (LAZ), decode them in parallel, validate the chunk table, and keep the serial path as the fallback and the test oracle.
6. Delay the loaders until the first frame is on screen.

Next: when something looks wrong in the pool, how do you tell? [Post 08](../post08-pool-order-and-upload-batches/).

## Where this lives in the code

`src/app/loading/ingest.cpp` (`start_load`, `spawn_loader`, `presize`, `ingest`), `src/app/loading/load_job.cpp` (`order_cloud`), `src/io/load.hpp`, `ply.cpp`, `las.cpp`, `src/core/parallel.hpp`. Numbers: ROADMAP 2.2, 2.2d, BENCH.md F5, FIXES F5.
