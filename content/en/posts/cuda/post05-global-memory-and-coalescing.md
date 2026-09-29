---
title: "Post 05: Global memory and coalescing"
date: 2026-09-29T19:05:00+03:00
weight: 6
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 03, we turned kernel times into GB/s and saw that a simple vector add reaches more than 90% of the RTX 4090's 1008 GB/s. In post 04, we pointed `ncu` at that kernel and saw that it spends its time waiting for memory. So for kernels like these, memory bandwidth is the whole game.

In this post, we will see that bandwidth is not just about *how many* bytes a kernel reads, but about *which* bytes each thread reads. We will write two kernels that copy exactly the same amount of data, with the same number of threads, and one of them will be five times slower than the other. The only difference is the addresses the threads touch.

## What We Will Learn

In this article, we will cover:

* How the loads of a warp turn into memory transactions: 32-byte sectors and 128-byte cache lines.
* What *coalesced* access means, and what strided access costs.
* Why a naive matrix transpose is slow, and whether it is better to make its reads or its writes coalesced.
* How to check all of this with `ncu` metrics instead of guessing.

## Table of Contents

1. The Full Program.
2. How a Warp Reads Memory.
3. Strided Copy.
4. Naive Matrix Transpose.
5. Build and Run.
6. Checking with ncu.
7. Exercise.

---

## 1. The Full Program

```bash
mkdir coalescing
cd coalescing
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post05/main.cu" >}}

`CUDA_CHECK`, `make_device_buffer`, and `time_ms` are the same as before. `time_ms` runs the kernel 3 times to warm up, then 20 times between two CUDA events, and returns the average. `peak_gbs` is the peak bandwidth of the RTX 4090 that we computed in post 03. If you have a different GPU, put your own number there.

---

## 2. How a Warp Reads Memory

In post 02, we saw that the GPU runs threads in groups of 32 called *warps*, and all 32 threads of a warp execute the same instruction at the same time. That includes loads. When a warp executes `x = in[i]`, the hardware receives 32 addresses at once, one per thread.

The memory system does not fetch those 32 floats one by one. It moves data in fixed-size chunks:

* A **sector** is 32 bytes. This is the smallest piece the GPU reads from or writes to global memory. Asking for one `float` (4 bytes) still moves the whole 32-byte sector that contains it.
* A **cache line** is 128 bytes, which is 4 sectors in a row.

So for each load instruction of a warp, the hardware looks at the 32 addresses and works out which sectors they fall in. Then it fetches each of those sectors once. `ncu` calls one warp-wide load instruction a *request*, and counts the *sectors* each request needs.

Now take the best case. Thread 0 reads `in[0]`, thread 1 reads `in[1]`, and so on up to thread 31. That is 32 floats × 4 bytes = 128 consecutive bytes, which is exactly 4 sectors, or one cache line. Every byte that comes from memory is used by some thread. This is a **coalesced** access: the 32 small requests of the warp merge into a few large ones.

Now say each thread reads with a stride: thread `t` reads `in[t * stride]`.

| stride | bytes between threads | sectors per request | bytes moved per useful byte |
| ------ | --------------------- | ------------------- | --------------------------- |
| 1      | 4                     | 4                   | 1×                          |
| 2      | 8                     | 8                   | 2×                          |
| 4      | 16                    | 16                  | 4×                          |
| 8      | 32                    | 32                  | 8×                          |
| 16     | 64                    | 32                  | 8×                          |
| 32     | 128                   | 32                  | 8×                          |

With stride 2, every sector holds only 4 floats the warp wants, so the warp needs twice as many sectors for the same 32 floats. With a stride of 8 or more, each thread lands in its own sector. The warp moves 32 × 32 = 1024 bytes to use 128 of them. It cannot get worse than that, because a sector is the smallest unit: 32 sectors is the maximum for 32 threads.

The GPU memory keeps running at full speed in all of these cases. It is just busy moving bytes nobody asked for. Let's measure it.

---

## 3. Strided Copy

{{< code file="cuda/post05/main.cu" region="strided" link="false" >}}

Each thread writes one element of `out`, and neighbouring threads write neighbouring elements, so the writes are always coalesced. Only the read changes with `stride`. We cast `i` to `long long` before multiplying because `i * stride` goes above 2^31 for the larger strides.

{{< code file="cuda/post05/main.cu" region="strided-main" link="false" >}}

Every launch copies the same 32M floats (128 MB), whatever the stride. For stride 32, the input has to be 32 times bigger, so we allocate 4 GB for it. You need about 5 GB of free GPU memory to run this program. If you have less, lower `max_stride` to 16.

The sizes are chosen on purpose. The RTX 4090 has a 72 MB L2 cache. If our buffers were smaller than that, the 20 timed runs would find the data waiting in L2 from the previous run, and we would measure the cache instead of the memory. With 128 MB read and 128 MB written per launch, the data cannot stay in L2 between runs.

As in post 03, we count only the *useful* bytes, `n` floats read plus `n` floats written. That is the number that matters to the user of the kernel, and it is why this is called *effective* bandwidth.

Output:

```text
stride  1: 0.291 ms,  922 GB/s (91.5% of peak)
stride  2: 0.433 ms,  620 GB/s (61.5% of peak)
stride  4: 0.718 ms,  374 GB/s (37.1% of peak)
stride  8: 1.298 ms,  207 GB/s (20.5% of peak)
stride 16: 1.296 ms,  207 GB/s (20.6% of peak)
stride 32: 1.431 ms,  188 GB/s (18.6% of peak)
```

Stride 1 is as fast as the vector add from post 03. From there, each doubling of the stride costs a lot, until stride 8, and then it stops getting worse. That is the shape the sector table predicts.

We can check the table directly. One pass over 32M floats is 134 MB, counted in units of 10^6 bytes like our GB/s. Take stride 4. The table says reads move 4 times the useful bytes, so one launch moves 4 × 134 MB of reads plus 134 MB of writes, which is 671 MB in 0.718 ms, or about 935 GB/s. For stride 8, it is 8 × 134 MB + 134 MB = 1208 MB in 1.298 ms, about 931 GB/s. For stride 2, about 930 GB/s. The memory is working at full speed in every row. The strided kernels are not slow because the memory is slow. They are slow because they spend that bandwidth on sectors they throw away.

Stride 32 is a little slower again: it moves the same number of sectors as stride 8, but now every thread also touches a different 128-byte cache line. We will not chase that last 10% here; the big lesson is the factor of 4.5 between stride 1 and stride 8.

---

## 4. Naive Matrix Transpose

Strided access is not something only artificial benchmarks do. Any time a kernel walks down a *column* of a row-major matrix, neighbouring threads read addresses one full row apart. The classic example is the matrix transpose: `out[x][y] = in[y][x]`. Whatever we do, one side of that assignment walks along a row and the other walks down a column.

Our matrix is 8192 × 8192 floats, which is 256 MB per matrix, far bigger than L2.

{{< code file="cuda/post05/main.cu" region="transpose-setup" link="false" >}}

### A baseline: copy

Before transposing, we want to know how fast a kernel with this shape *could* be. `copy_2d` uses the same blocks and indexing as the transposes, but copies the matrix without transposing it:

{{< code file="cuda/post05/main.cu" region="copy2d" link="false" >}}

Each block handles a 32 × 32 tile of the matrix with 32 × 8 = 256 threads, so each thread handles 4 rows of the tile, at `y`, `y + 8`, `y + 16`, and `y + 24`. `threadIdx.x` goes along a row, so for one warp, `x` takes 32 consecutive values and `y` is the same for all threads. Both the read and the write of a warp cover 32 consecutive floats: fully coalesced.

We use `TILE` and `ROWS` here because they are the same shape we will need in the next post, where each block will load its tile into shared memory.

### Two ways to transpose

{{< code file="cuda/post05/main.cu" region="transpose" link="false" >}}

The two kernels do the same work and produce the same result. They only swap which side is indexed along a row:

* `transpose_read_coalesced` reads `in[(y + j) * n + x]`. For a warp, `x` changes from thread to thread, so the read is 32 consecutive floats. But the write goes to `out[x * n + ...]`, and consecutive `x` are `n` floats apart, which is 32 KB. Every thread of the warp writes to a different sector.
* `transpose_write_coalesced` does the opposite: coalesced writes, and reads that jump by 32 KB from thread to thread.

So each kernel has one perfect access (4 sectors per request) and one terrible access (32 sectors per request).

{{< code file="cuda/post05/main.cu" region="transpose-run" link="false" >}}

We loop over the three kernels with a small table of function pointers, and check every element of the result on the host after each one. Here we count 2 × 256 MB of useful bytes per launch: one full matrix read and one full matrix written.

Output:

```text
copy_2d: 0.582 ms, 922 GB/s (91.5% of peak) PASSED
transpose_read_coalesced: 1.739 ms, 309 GB/s (30.6% of peak) PASSED
transpose_write_coalesced: 1.426 ms, 376 GB/s (37.4% of peak) PASSED
```

The copy runs at 91.5% of peak, the same as our best strided copy. Both transposes are around three times slower, even though they move exactly the same data. Half of their memory accesses are uncoalesced, and that is enough to lose most of the bandwidth.

The interesting part is that the two transposes are *not* equal. Coalescing the writes is faster than coalescing the reads. Why?

It comes down to how the caches treat loads and stores:

* **Strided reads can be rescued by the cache.** In `transpose_write_coalesced`, a warp reads 32 floats down one column, so it pulls in 32 sectors and uses one float of each. But the other 7 floats in each of those sectors belong to the next 7 columns, and those are exactly what the other warps of the same block read, a moment later. The first warp to touch a sector brings it into the cache, and the others find it there. Much of the waste is paid once, not eight times.
* **Strided writes cannot.** The L1 cache of each SM does not keep global stores. Every store goes on to L2, and a warp whose 32 threads write to 32 different sectors sends 32 separate small writes, each covering only 4 bytes of its sector. Nothing merges them on the way.

So if you must choose which side of a kernel to leave uncoalesced, leave the reads. But the real answer is to not choose at all. We need both sides coalesced, and that requires the threads of a block to cooperate: read a tile along rows, exchange the data, and write it along rows. That is what shared memory is for.

---

## 5. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o coalescing
./coalescing
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post05/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/coalescing
```
{{< /tab >}}
{{< /tabs >}}

Output:

```text
stride  1: 0.291 ms,  922 GB/s (91.5% of peak)
stride  2: 0.433 ms,  620 GB/s (61.5% of peak)
stride  4: 0.718 ms,  374 GB/s (37.1% of peak)
stride  8: 1.298 ms,  207 GB/s (20.5% of peak)
stride 16: 1.296 ms,  207 GB/s (20.6% of peak)
stride 32: 1.431 ms,  188 GB/s (18.6% of peak)

copy_2d: 0.582 ms, 922 GB/s (91.5% of peak) PASSED
transpose_read_coalesced: 1.739 ms, 309 GB/s (30.6% of peak) PASSED
transpose_write_coalesced: 1.426 ms, 376 GB/s (37.4% of peak) PASSED
```

Your numbers will differ on another GPU, but the shape should be the same: stride 1 and the copy near the peak, a big drop up to stride 8, and write-coalesced beating read-coalesced.

---

## 6. Checking with ncu

So far, the sector story is a theory that happens to fit our timings. `ncu` can count the sectors directly. These are the metrics we want:

| Metric | Meaning |
| ------ | ------- |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Number of warp-wide global load instructions (requests) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Number of 32-byte sectors those loads needed |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | Sectors per request: 4 is perfect for `float` |
| `..._op_st` versions of the three above | The same, for global stores |

You can list every metric your GPU supports with `ncu --query-metrics`.

Each kernel launch in our program is repeated 23 times (3 warm-up runs and 20 timed ones), and we only need one of them. `-k` picks the kernel by name, `-c 1` profiles one launch, and `--launch-skip` skips the launches before it. So to see all six strides:

```bash
for k in 0 1 2 3 4 5; do
  ncu -k strided_copy --launch-skip $((23 * k)) -c 1 \
      --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
      ./build/coalescing
done
```

And for the two transposes, loads and stores side by side, plus the **Memory Workload Analysis** tables:

```bash
for kernel in transpose_read_coalesced transpose_write_coalesced; do
  ncu -k $kernel -c 1 --section MemoryWorkloadAnalysis_Tables \
      --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
      ./build/coalescing
done
```

If `ncu` stops with `ERR_NVGPUCTRPERM`, your user is not allowed to read the GPU performance counters. We covered the fix in post 04: either run it once as root with the full path, `sudo /usr/local/cuda/bin/ncu ...`, or allow all users by adding `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to a file in `/etc/modprobe.d/` and rebooting.

Here is what to look for. The numbers below are not a pasted `ncu` report. They follow from the sector arithmetic of section 2, so if your counters disagree, it is worth finding out why.

For the strided copy, with `n` = 32M floats, each launch issues 32M / 32 = 1,048,576 load requests, whatever the stride. The sectors grow with the stride:

| stride | load requests | load sectors | sectors per request |
| ------ | ------------- | ------------ | ------------------- |
| 1      | 1,048,576     | 4,194,304    | 4                   |
| 2      | 1,048,576     | 8,388,608    | 8                   |
| 4      | 1,048,576     | 16,777,216   | 16                  |
| 8      | 1,048,576     | 33,554,432   | 32                  |
| 16     | 1,048,576     | 33,554,432   | 32                  |
| 32     | 1,048,576     | 33,554,432   | 32                  |

For the transposes, the two kernels should mirror each other:

| kernel | sectors per load request | sectors per store request |
| ------ | ------------------------ | ------------------------- |
| `transpose_read_coalesced` | 4 | 32 |
| `transpose_write_coalesced` | 32 | 4 |

The counts alone do not explain why write-coalesced is faster, since both kernels have one bad side. The Memory Workload Analysis tables do. Look at the **L1/TEX hit rate** for global loads: for `transpose_write_coalesced`, a large share of those 32 sectors per request should be hits, because other warps of the block already brought them in. That is the cache rescuing the strided reads, as described in section 4. `ncu` also prints a hint under the table when only a few of the 32 bytes of each sector are used by the threads; that hint is the word "uncoalesced" in `ncu`'s own language.

Remember from post 04 that `ncu` flushes the caches and locks the clocks before profiling, so its `Duration` will not match our `cudaEvent` times. Use `ncu` for the counts and the reasons, and `time_ms` for the score.

---

## 7. Exercise

Coalescing needs neighbouring threads to read neighbouring addresses. Does it also need the warp to start at the beginning of a sector?

Change the strided copy to read with an offset instead of a stride:

```cpp
out[i] = in[i + offset];
```

and run it for every `offset` from 0 to 32. With an offset of 1, each warp's 128 bytes start 4 bytes into a sector, so the warp needs 5 sectors instead of 4. You might expect a 25% slowdown.

Run it, and then check the sectors per request for `offset = 0` and `offset = 1` with `ncu`. Then explain why the time barely moves. (On my RTX 4090, every offset from 0 to 32 took the same 0.291 ms.) Hint: the fifth sector of one warp is the first sector of the next warp, and L2 sits between them and the memory.

In summary, a warp's memory access is only as good as the sectors it needs. The GPU moves global memory in 32-byte sectors, so 32 threads reading 32 consecutive floats cost 4 sectors, while 32 threads reading scattered floats cost 32. In our strided copy, the memory ran at full speed the whole time, and the effective bandwidth still fell from 922 GB/s to under 210 GB/s, because most of each sector was thrown away. A naive transpose cannot avoid having one uncoalesced side, and it runs at about a third of the copy's speed. Uncoalesced reads hurt a little less than uncoalesced writes, because the cache can reuse the extra sectors. In the next post, we will stop choosing between them: each block will load its 32 × 32 tile into shared memory with coalesced reads, and write it back out with coalesced writes, and we will see how close to the copy baseline that gets us.

[1]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory "CUDA C++ Best Practices Guide: Coalesced Access to Global Memory"
[2]: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/ "How to Access Global Memory Efficiently in CUDA C/C++ Kernels"
[3]: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ "An Efficient Matrix Transpose in CUDA C/C++"
[4]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "Nsight Compute Profiling Guide"
