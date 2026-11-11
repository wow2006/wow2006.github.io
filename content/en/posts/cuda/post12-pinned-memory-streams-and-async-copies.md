---
title: "Post 12: Pinned memory, streams & async copies"
date: 2026-09-29T19:12:00+03:00
weight: 13
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

Since [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}), every program in this series has followed the same three steps: copy in, run the kernel, copy out. Each step waits for the one before it. In post 04, the `nsys` timeline showed what that costs: while the data crosses PCIe, the GPU sits idle, and while the kernel runs, the PCIe link sits idle. We also saw that our copies from a `std::vector` only reached about 10 to 12 GB/s, and we promised to fix that here.

In this post, we will fix both problems. First, we will make the copies faster with *pinned* memory. Then, we will split the work into chunks and use *streams*, so that the GPU computes one chunk while the next one is still being copied.

## What We Will Learn

In this article, we will cover:

* What pinned memory is, and how much faster it copies than normal memory.
* How to allocate pinned memory with `cudaMallocHost` and free it with a `unique_ptr`.
* What a CUDA stream is, and how `cudaMemcpyAsync` puts a copy into one.
* How to split a large job into chunks, so copies and kernels run at the same time.
* How to see the overlap on the `nsys` timeline.

## Table of Contents

1. The Full Program.
2. Pageable vs. Pinned Memory.
3. Streams and Async Copies.
4. The Pipeline.
5. Timing Many Streams.
6. Build and Run.
7. Looking at the Timeline.
8. Exercise.

---

## 1. The Full Program

```bash
mkdir streams
cd streams
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post12/main.cu" >}}

We keep `CUDA_CHECK` and `DeviceBuffer` from post 01, and `time_ms` from post 03. The program does two things. It measures how fast a 256 MiB copy is from normal memory and from pinned memory. Then it runs the same workload with 1 stream and with 4 streams and compares the time.

---

## 2. Pageable vs. Pinned Memory

Memory you get from `new`, `malloc`, or `std::vector` is *pageable*. The operating system is free to move it around in physical RAM, or even write it out to swap. That is fine for the CPU, but the GPU copies data with its own DMA engine, which reads physical addresses directly over PCIe. If the OS moved a page in the middle of a copy, the GPU would read the wrong data.

So the driver cannot copy pageable memory directly. It first copies your data, piece by piece, into a small buffer that the OS is not allowed to move, and then the GPU copies from that buffer. Every byte is copied twice, and the CPU does half the work.

Memory the OS is not allowed to move is called *pinned* (or *page-locked*) memory. If our data already lives in pinned memory, the GPU can read it directly. We allocate it with `cudaMallocHost` and free it with `cudaFreeHost`. That is the same pattern as `cudaMalloc` and `cudaFree`, so the same `unique_ptr` trick from post 01 works:

{{< code file="cuda/post12/main.cu" region="pinned" link="false" >}}

We allocate the input and output on the host as pinned buffers, plus a normal `std::vector` for comparison:

{{< code file="cuda/post12/main.cu" region="alloc" link="false" >}}

The only difference is the allocator. We can use the pinned buffer like any other array, and `in.get()[i]` works on the CPU as usual.

Now we time one 256 MiB host-to-device copy from each:

{{< code file="cuda/post12/main.cu" region="bandwidth" link="false" >}}

We use `time_ms` from post 03 with one warm-up copy and 5 timed copies.

```text
H2D 256 MiB, pageable:  25.44 ms,  10.6 GB/s
H2D 256 MiB, pinned:    11.54 ms,  23.3 GB/s
```

The same copy is more than twice as fast from pinned memory. 23.3 GB/s is close to what PCIe 4.0 x16 can do in practice (the theoretical peak is about 32 GB/s, and protocol overhead eats a good part of it). The pageable number, 10.6 GB/s, matches what we measured in post 04.

Why not pin everything, then? Because pinned memory is taken away from the OS. It cannot be swapped out, so if you pin too much, the rest of the system runs short of memory. `cudaMallocHost` is also much slower than `malloc`, so allocate pinned buffers once at startup and reuse them. Pin the buffers that you copy to and from the GPU often, and nothing else.

---

## 3. Streams and Async Copies

Faster copies are nice, but the GPU still does one thing at a time: copy, then compute, then copy. To do better, we need to tell CUDA which operations depend on each other.

A *stream* is a queue of GPU work. Operations in the same stream run in order, one after the other. Operations in *different* streams have no order between them, so the GPU is free to run them at the same time.

We have been using a stream all along without knowing it. Every `cudaMemcpy` and every `<<<...>>>` launch so far went into the *default stream*. To use more than one, we create them with `cudaStreamCreate` and destroy them with `cudaStreamDestroy`. A `cudaStream_t` is a pointer to an opaque struct called `CUstream_st`, so `unique_ptr` handles it too:

{{< code file="cuda/post12/main.cu" region="stream" link="false" >}}

To put work into a stream, we pass it as an extra argument:

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(...);
```

`cudaMemcpyAsync` is `cudaMemcpy` with a stream at the end. The important word is *async*: it only puts the copy into the queue and returns right away. The CPU does not wait for the copy to finish. The kernel launch has two extra launch parameters: the third one is the amount of dynamic shared memory (we use none, so `0`), and the fourth is the stream.

There is one catch. `cudaMemcpyAsync` is only truly asynchronous when the host memory is pinned. With pageable memory, the driver has to stage the data through its own pinned buffer using the CPU, so the call behaves much like a plain `cudaMemcpy`. Pinned memory is not only faster, it is what makes overlap possible at all.

What can actually run at the same time? The RTX 4090 has two copy engines (`asyncEngineCount = 2` in `cudaDeviceProp`), so it can copy to the device, copy from the device, and run a kernel, all at once.

---

## 4. The Pipeline

For the overlap to matter, the kernel has to take about as long as the copies. A vector add is far too cheap: it finishes long before its data arrives. So we use a kernel that does real work for each element, 300 steps of `v = sin(v) + 1`:

{{< code file="cuda/post12/main.cu" region="kernel" link="false" >}}

`crunch` is marked `__host__ __device__`, so the same function can be called from the kernel and from the CPU. We use that later to check the result.

Now the key idea. We have 64M floats (256 MiB) to process. Instead of copying all of them, computing all of them, and copying all of them back, we split them into 16 chunks of 4M floats. Each chunk gets its own copy-in, kernel, and copy-out, and we give the chunks to our streams in turn: chunk 0 to stream 0, chunk 1 to stream 1, and so on, with chunk 4 going back to stream 0.

{{< code file="cuda/post12/main.cu" region="pipeline" link="false" >}}

Inside one stream, the three operations of a chunk stay in order: the kernel waits for its copy-in, and the copy-out waits for the kernel. That is exactly the dependency we need, and we did not have to write any synchronization for it. Between streams, there is no order. So while stream 0 runs the kernel for chunk 0, stream 1 is free to copy in chunk 1. After a short start, the copy engines and the SMs are all busy at the same time.

Each chunk uses its own slice of the device buffer (`dev + offset`), so chunks in different streams never touch the same memory. If two streams shared one device buffer, chunk 1's copy-in could overwrite chunk 0's data while chunk 0's kernel is still reading it.

Notice that `run_pipeline` does not wait for anything. It only puts 48 operations into the queues and returns almost immediately. The GPU does the actual work later.

---

## 5. Timing Many Streams

{{< code file="cuda/post12/main.cu" region="streams" link="false" >}}

We run the same pipeline with 1 stream and with 4 streams. With 1 stream, all 48 operations are in one queue, so they run strictly one after another, just like our earlier programs. With 4 streams, they are allowed to overlap.

How can `time_ms` measure this? It records its events into the default stream, not into our streams. It works because streams created with `cudaStreamCreate` are *blocking* streams: the default stream waits for all work already queued in them, and work queued in them after an event waits for that event. So the `stop` event completes only after every chunk in every stream is done.

Finally, we check the result. Running `crunch` 64 million times on the CPU would be slow, so we check every 997th element, which still covers every chunk:

{{< code file="cuda/post12/main.cu" region="verify" link="false" >}}

The GPU's `sinf` and the CPU's `sinf` are not bit-for-bit identical, so we compare with a small tolerance instead of `!=`.

---

## 6. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o streams
./streams
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post12/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/streams
```
{{< /tab >}}
{{< /tabs >}}

Output (RTX 4090, PCIe 4.0 x16):

```text
H2D 256 MiB, pageable:  25.44 ms,  10.6 GB/s
H2D 256 MiB, pinned:    11.54 ms,  23.3 GB/s
1 stream(s), 16 chunks:  38.38 ms
4 stream(s), 16 chunks:  24.24 ms
PASSED (0 errors)
```

The same work, the same kernel, the same data: 38.4 ms with one stream, 24.2 ms with four. That is 1.6 times faster, and we did not touch the kernel at all.

Where do the numbers come from? With one stream, the time is simply the sum of the parts: about 12 ms to copy in, 14.5 ms of kernels, and 12 ms to copy out. With four streams, the time is about 24 ms, which is the two copies together. The kernel time has disappeared completely: it now runs in the shadow of the copies. The next section shows why the copies themselves did not overlap as well.

---

## 7. Looking at the Timeline

Let's record the run with `nsys`, as in post 04:

```bash
nsys profile -t cuda -o streams ./streams
nsys stats -q --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum streams.nsys-rep
```

```text
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)         Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------
    100.0      177,453,286        192  924,235.9  909,063.0   900,199   983,559     29,459.6  heavy(float *, int)

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count   Avg (ns)    Med (ns)   Min (ns)   Max (ns)   StdDev (ns)           Operation
 --------  ---------------  -----  -----------  ---------  --------  ----------  -----------  ----------------------------
     75.0      435,237,724    204  2,133,518.3  767,349.5   685,286  26,488,238  4,494,860.7  [CUDA memcpy Host-to-Device]
     25.0      145,070,552    192    755,575.8  738,806.0   668,165     982,311     60,097.3  [CUDA memcpy Device-to-Host]
```

The summary tells us each kernel takes about 0.9 ms and each 16 MB chunk copy about 0.75 ms. (The 26 ms maximum is one of the big 256 MiB pageable copies from the bandwidth test.) But a summary adds times up; it cannot show overlap. For that, we need the trace:

```bash
nsys stats -q --report cuda_gpu_trace streams.nsys-rep
```

The real table has many columns. Below, we keep only the start time, duration, throughput, stream, and name, and shift the start times so each excerpt begins at 0. All times are in milliseconds.

First, a slice from the 1-stream run:

```text
    Start   Duration  Throughput (MB/s)  Strm  Name
    0.000      0.712         23,538.434    14  [CUDA memcpy Host-to-Device]
    0.722      0.901                       14  heavy(float *, int)
    1.631      0.815         20,568.867    14  [CUDA memcpy Device-to-Host]
    2.449      0.720         23,286.776    14  [CUDA memcpy Host-to-Device]
    3.172      0.901                       14  heavy(float *, int)
    4.078      0.847         19,813.892    14  [CUDA memcpy Device-to-Host]
    4.936      0.734         22,850.568    14  [CUDA memcpy Host-to-Device]
```

Each operation starts when the previous one ends. In the Nsight Systems GUI (`nsys-ui streams.nsys-rep`), this looks like one row of blocks for stream 14, laid end to end like a staircase: a copy block, a kernel block, a copy block, and so on. Nothing is ever stacked on top of anything else.

Now a slice from the middle of the 4-stream run. Our four streams show up as streams 15 to 18:

```text
    Start   Duration  Throughput (MB/s)  Strm  Name
    0.000      0.758         22,129.148    18  [CUDA memcpy Host-to-Device]
    0.759      0.759         22,112.371    17  [CUDA memcpy Host-to-Device]
    0.766      0.907                       18  heavy(float *, int)
    1.519      2.139          7,834.960    15  [CUDA memcpy Host-to-Device]
    1.638      0.921                       17  heavy(float *, int)
    1.678      0.732         22,900.900    18  [CUDA memcpy Device-to-Host]
    2.563      0.723         23,186.113    17  [CUDA memcpy Device-to-Host]
    3.660      0.757         22,162.702    16  [CUDA memcpy Host-to-Device]
    3.664      0.906                       15  heavy(float *, int)
    4.418      2.131          7,868.514    18  [CUDA memcpy Host-to-Device]
    4.536      0.921                       16  heavy(float *, int)
    4.572      0.708         23,689.429    15  [CUDA memcpy Device-to-Host]
    5.466      0.711         23,605.543    16  [CUDA memcpy Device-to-Host]
```

Read it line by line:

* At 0.759 ms, stream 18's kernel starts on the chunk that just arrived, and at the same moment, stream 17 starts copying in the next chunk. This is the overlap we wanted: copy of chunk N+1 during the compute of chunk N.
* At 1.678 ms, stream 18 copies its result back while stream 17's kernel is still running and stream 15 is copying in. Three operations run at once: one per copy engine plus the kernel.
* Look at the throughput column. A copy-in that runs alone gets about 22 GB/s. A copy-in that runs at the same time as a copy-out (at 1.519 and 4.418 ms) drops to 7.8 GB/s and takes almost three times as long. The copy-out next to it keeps its full 23 GB/s.

In the GUI, the 4-stream section looks very different from the staircase. Under the CUDA HW row you see four stream rows, and their blocks are shifted against each other, so at almost any point in time a vertical line crosses two or three blocks. The kernel row is busy in pairs of 0.9 ms blocks, with small gaps while it waits for a slow copy-in to finish.

That last bullet explains our 24 ms. On this machine, copies to and from the GPU do not both run at full speed at the same time: they share about 23 GB/s between them. So the 256 MiB in plus 256 MiB out take about 24 ms no matter how we arrange them, and that is the floor for this program. Our 4-stream run is at 24.2 ms, so it has reached it. The kernels, 14.5 ms in total, are completely hidden. On a system where both directions can run at full speed at the same time, the floor would be the largest single part (about 14.5 ms here) instead.

This is the general rule of overlap: **the total time becomes the time of the busiest resource**, not the sum of all of them. We can check it by making the kernel heavier. With `kIters = 800`, each chunk's kernel takes longer than its copies, and we measured:

```text
1 stream(s), 16 chunks:  62.33 ms
4 stream(s), 16 chunks:  38.79 ms
```

Now the kernels are the busiest resource (about 38 ms in total) and the 4-stream time is almost exactly that. This time, the copies are the part that disappears.

---

## 8. Exercise

Run the pipeline from pageable memory instead of pinned memory. Add two `std::vector`s before the `streams` region:

```cpp
std::vector<float> pg_in(in.get(), in.get() + n), pg_out(n);
```

pass `pg_in.data()` and `pg_out.data()` to `run_pipeline` instead of `in.get()` and `out.get()`, and check `pg_out` in the verify loop. Before you run it, predict both times.

Here is what we measured:

```text
1 stream(s), 16 chunks:  66.14 ms
4 stream(s), 16 chunks:  67.45 ms
PASSED (0 errors)
```

Two things happened. Both runs are much slower than the pinned 1-stream run, because every copy now goes at pageable speed. And 4 streams are no faster than 1: the driver has to stage every pageable copy through its own buffer with the CPU, so `cudaMemcpyAsync` no longer returns right away, and the overlap is gone. Record it with `nsys` and look at the trace. Even with 4 streams, it is a staircase again: every copy and every kernel starts only after the previous one has ended, and the `SrcMemKd`/`DstMemKd` columns say `Pageable` instead of `Pinned`.

As a second experiment, change `chunks` to 4 and to 64 and run the pinned version again. We measured 22.8 ms (4 chunks) and 23.5 ms (64 chunks) with 4 streams, so on this machine the chunk count barely matters: the shared PCIe bandwidth sets the floor, not the pipeline shape. Try it with `kIters = 800` too, where the kernels are the busiest resource, and see whether that changes.

In summary, pinned memory from `cudaMallocHost` copies more than twice as fast as pageable memory, and it is required for `cudaMemcpyAsync` to be truly asynchronous. A stream is a queue: work inside one stream runs in order, and work in different streams may run at the same time. By splitting a large job into chunks and spreading them over a few streams, the GPU copies one chunk while it computes another. The total time then drops from the sum of all the parts to the time of the busiest one, which here took our program from 38.4 ms to 24.2 ms without changing the kernel. `nsys` shows the overlap directly, and it also shows the limits of your machine, like our copy-in and copy-out sharing the PCIe bandwidth.

[1]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation "CUDA C++ Best Practices Guide: Asynchronous and Overlapping Transfers with Computation"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html "CUDA Runtime API: Stream Management"
[3]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html "CUDA Runtime API: Memory Management"
[4]: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ "How to Optimize Data Transfers in CUDA C/C++"
[5]: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/ "How to Overlap Data Transfers in CUDA C/C++"
[6]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "Nsight Systems User Guide"
