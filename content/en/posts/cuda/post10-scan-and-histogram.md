---
title: "Post 10: Scan and histogram"
date: 2026-09-29T19:10:00+03:00
weight: 11
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 09, we reduced an array to a single sum. Every thread worked on its own piece, and the partial results were combined at the end with a tree, warp shuffles, and a few atomics. In this post, we will use the same tools on two other patterns that show up everywhere in GPU code: the *prefix sum* (or *scan*) and the *histogram*.

A scan is a reduction that keeps every intermediate result. A histogram is many small reductions at once, one per bin. Both look simple on the CPU, where a single loop does the job, but both need a new idea on the GPU.

## What We Will Learn

In this article, we will cover:

* What an inclusive scan is and how one block scans 1024 numbers.
* How to scan a warp with `__shfl_up_sync`.
* How to scan an array of any size with several blocks.
* Why a histogram with global atomics is slow, and how per-block copies (privatization) fix it.
* How our kernels compare with `cub::DeviceScan` and `cub::DeviceHistogram`.

## Table of Contents

1. What Is a Scan?
2. The Scan Program.
3. Scanning One Block in Shared Memory.
4. Scanning One Block with Warp Shuffles.
5. Scanning Many Blocks.
6. The Histogram Program.
7. Histogram with Global Atomics.
8. Privatization: One Histogram per Block.
9. Build and Run.
10. Exercise.

---

## 1. What Is a Scan?

An *inclusive scan* replaces every element with the sum of itself and everything before it:

```text
input:  3  1  4  1  5  9  2  6
output: 3  4  8  9 14 23 25 31
```

The C++ standard library already has it as `std::inclusive_scan`. There is also an *exclusive* scan, where each output leaves out its own element (`0 3 4 8 9 14 23 25`). In this post, we only do the inclusive one.

Why is it useful? A scan answers the question "where does my output go?". Suppose each thread wants to write a different number of results into one packed array. If we scan the counts, each thread gets its starting position. This is how GPUs remove elements from an array (stream compaction), build sparse matrices, and implement radix sort.

On the CPU, a scan is a loop where each step depends on the previous one. That looks impossible to parallelize. The trick is to do a bit more work in exchange for steps that can run at the same time.

---

## 2. The Scan Program

```bash
mkdir scan-histogram
cd scan-histogram
code scan.cu
```

This post has two programs. We start with the scan. You can write the following in `scan.cu`

{{< code file="cuda/post10/scan.cu" >}}

We keep `CUDA_CHECK`, the templated `DeviceBuffer` from post 09, and `time_ms` from post 03. The input is 30 million random digits from 0 to 9, and we compute the expected result with `std::inclusive_scan`:

{{< code file="cuda/post10/scan.cu" region="host" link="false" >}}

The input is 120 MB, larger than the 72 MB L2 cache, so we measure real memory bandwidth. We use `int` instead of `float` so the GPU result must match the CPU result exactly. The last element is about 135 million, which fits comfortably in an `int`.

---

## 3. Scanning One Block in Shared Memory

We first solve a smaller problem: scan 1024 numbers with one block of 1024 threads, one number per thread.

{{< code file="cuda/post10/scan.cu" region="hillis-steele" link="false" >}}

This is the *Hillis-Steele* scan. Each thread loads one element into shared memory. Then, in every step, each thread adds the value `offset` places to its left, and `offset` doubles each time:

```text
start:       3  1  4  1  5  9  2  6
offset = 1:  3  4  5  5  6 14 11  8
offset = 2:  3  4  8  9 11 19 17 22
offset = 4:  3  4  8  9 14 23 25 31
```

After step `k`, each element holds the sum of the `2^k` elements ending at it. For 1024 elements, we need 10 steps instead of 1023.

Notice the two `__syncthreads()` calls in the loop. Every thread reads a neighbour's value and then writes its own. Without the first barrier, a thread could overwrite its value before its neighbour on the right has read it. This is exactly the kind of race we caught with `compute-sanitizer --tool racecheck` in post 06.

Each thread also writes the block's total, the last element, into `block_sums`. We will need it in section 5.

---

## 4. Scanning One Block with Warp Shuffles

In post 09, we summed a warp with `__shfl_down_sync`, without touching shared memory. A scan can do the same with `__shfl_up_sync`, which reads a value from the lane `offset` places below:

{{< code file="cuda/post10/scan.cu" region="warp-scan" link="false" >}}

`warp_scan` is the same Hillis-Steele loop, but on 32 registers instead of shared memory. The lanes of a warp run together, so we need no barriers, and 5 steps are enough for 32 values. Lanes below `offset` have nobody to their left, so they keep their value.

A block of 1024 threads has 32 warps, so the kernel works in three steps:

1. Each warp scans its own 32 values. Lane 31 now holds the warp's total, and stores it in `warp_sums`.
2. The first warp scans the 32 warp totals. The same `warp_scan` function works, because there are exactly 32 of them.
3. Each thread adds the total of all the warps to its left, `warp_sums[warp - 1]`.

We need only two `__syncthreads()` here, instead of 21 in the shared-memory version.

---

## 5. Scanning Many Blocks

Our array has 30 million elements, far more than one block can handle. Each block can only scan its own 1024 elements, because blocks cannot wait for each other. So we split the work into three steps:

{{< code file="cuda/post10/scan.cu" region="multi-block" link="false" >}}

1. Every block scans its own 1024 elements and writes its total to `block_sums`.
2. We scan `block_sums`. After that, `block_sums[b]` holds the total of all blocks up to and including `b`.
3. Every element of block `b` adds `block_sums[b - 1]`, the total of all the blocks before it. Block 0 has nothing to add.

Step 2 is itself a scan, so `scan` calls itself. With 30 million elements, the first level has 29,297 blocks. Scanning those 29,297 block sums needs 29 blocks, and scanning those 29 sums needs a single block, where the recursion stops.

`scan` works with either block kernel, so we pass the kernel as a function pointer. The block sums of every level go into one scratch buffer that we allocate once, outside the timing loop:

{{< code file="cuda/post10/scan.cu" region="scratch" link="false" >}}

That is 29,297 + 29 + 1 = 29,327 ints. If we called `cudaMalloc` inside `scan`, we would be timing the allocations too.

We measure each version and compare it with the CPU result. We count the bytes a scan must move at least, reading `n` ints and writing `n` ints, and compare with a device-to-device copy of the same size, the ceiling from post 09:

{{< code file="cuda/post10/scan.cu" region="report" link="false" >}}

{{< code file="cuda/post10/scan.cu" region="run" link="false" >}}

Finally, CUB does the whole thing in one call. It uses the same two-call pattern as `cub::DeviceReduce` in post 09: the first call only asks how much temporary memory it needs.

{{< code file="cuda/post10/scan.cu" region="cub" link="false" >}}

We will look at CUB and the other libraries in the next post.

---

## 6. The Histogram Program

A histogram counts how many times each value appears. We count the bytes of 256 MiB of random data into 256 bins, one bin per byte value. Create `histogram.cu`

{{< code file="cuda/post10/histogram.cu" >}}

The CPU version is the loop at the end of the `host` region:

{{< code file="cuda/post10/histogram.cu" region="host" link="false" >}}

With random bytes, each bin should get about 268 million / 256 ≈ 1 million counts.

---

## 7. Histogram with Global Atomics

The obvious GPU version gives each thread some bytes and increments the right bin with `atomicAdd`:

{{< code file="cuda/post10/histogram.cu" region="global" link="false" >}}

We use a grid-stride loop from post 02, so a fixed number of blocks covers any size. `atomicAdd` is needed because many threads may increment the same bin at the same time. A plain `bins[data[i]]++` would lose updates, just like the unprotected sum in post 09.

The code is correct, but it is very slow. Every one of the 268 million bytes turns into an atomic operation in global memory, and all of them go to only 256 addresses. Atomics on the same address are handled one after another, so the whole GPU ends up waiting in a few queues.

---

## 8. Privatization: One Histogram per Block

The fix is to give each block its own private histogram in shared memory, and merge it into the global one only once at the end:

{{< code file="cuda/post10/histogram.cu" region="shared" link="false" >}}

The kernel has three phases, separated by `__syncthreads()`:

1. Clear the 256 shared counters.
2. Count this block's bytes with `atomicAdd` on shared memory. Shared-memory atomics are handled inside the SM, and only the threads of one block compete for each counter.
3. Add the block's 256 counters to the global histogram. We skip empty bins.

The number of global atomics drops from 268 million to at most 1024 blocks × 256 bins = 262,144. This trick is called *privatization*, and it works for any reduction where many threads update a few locations.

We launch 8 blocks per SM:

{{< code file="cuda/post10/histogram.cu" region="run" link="false" >}}

That is enough blocks to keep the GPU busy, and few enough that the merge in phase 3 stays small. Because each run adds to `bins`, we clear it with `cudaMemsetAsync` before every launch. The memset is part of the timed work, but 1 KB is nothing next to 256 MiB.

CUB has a ready-made version, `cub::DeviceHistogram::HistogramEven`. We ask for 257 *levels*, the boundaries between bins, evenly spaced from 0 to 256, which gives 256 bins of width 1:

{{< code file="cuda/post10/histogram.cu" region="cub" link="false" >}}

---

## 9. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 scan.cu -o scan
nvcc -U_GNU_SOURCE -arch=native -O3 histogram.cu -o histogram
./scan
./histogram
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to the two `.cu` files

{{< code file="cuda/post10/CMakeLists.txt" >}}

One project can build several programs; each `add_executable` line gives us one.

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/scan
./build/histogram
```
{{< /tab >}}
{{< /tabs >}}

We include only the CUB headers we use, `<cub/device/device_scan.cuh>` and `<cub/device/device_histogram.cuh>`. They compile with `-U_GNU_SOURCE` as they are. If you include the umbrella header `<cub/cub.cuh>` instead, it pulls in `<mutex>`, and you get the `pthread_cond_clockwait` errors we worked around in post 09.

Output of `./scan` (RTX 4090):

```text
n = 30000000 ints (120 MB), scratch = 29327 ints

copy (ceiling):          0.263 ms,  913 GB/s (90.6% of peak)
v1 shared Hillis-Steele: 0.684 ms,  351 GB/s (34.8% of peak) OK
v2 warp shuffle:         0.538 ms,  446 GB/s (44.2% of peak) OK
cub::DeviceScan:         0.262 ms,  915 GB/s (90.7% of peak) OK

last element = 134982752 (expected 134982752)
PASSED
```

The warp-shuffle version is about 27% faster than the shared-memory one. Fewer barriers and no shared-memory traffic in the inner loop make a difference.

But both are far from the copy ceiling, and the reason is our three-step design, not the block kernel. Step 1 reads and writes the whole array, and step 3 reads and writes it again. That is twice the traffic of a copy, so the best this design can reach is half of 913 GB/s, about 456 GB/s. At 446 GB/s, `v2` is already close to that limit.

CUB scans at copy speed. It uses a *single-pass* scan with *decoupled look-back* [1]: each block publishes its total as soon as it knows it, and the next block reads it directly instead of waiting for a second pass. Every element is read once and written once, just like a copy.

Output of `./histogram`:

```text
n = 268435456 bytes (268 MB), 256 bins

v1 global atomics:       57.056 ms,    5 GB/s (0.5% of peak) OK
v2 shared atomics:       0.555 ms,  484 GB/s (48.0% of peak) OK
cub::DeviceHistogram:    0.285 ms,  940 GB/s (93.3% of peak) OK

grid: 1024 blocks x 256 threads
bins[0] = 1049260, bins[255] = 1047898
PASSED
```

Privatization makes the histogram 100 times faster, with the same result. The global atomics version reaches 0.5% of the memory bandwidth; the GPU spends almost all its time waiting for atomics, not for memory.

Our shared version is still at half of CUB's speed. The atomics are no longer the problem; the loads are. Each thread reads a single byte per iteration, which is too little data per memory request. The exercise fixes that.

---

## 10. Exercise

Change the counting loop in `histogram_shared` so that each thread reads 4 bytes at once as a `uchar4`, then does 4 shared-memory atomics:

```cpp
const uchar4* data4 = reinterpret_cast<const uchar4*>(data);
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += blockDim.x * gridDim.x) {
    uchar4 v = data4[i];
    atomicAdd(&local[v.x], 1u);
    // ... and v.y, v.z, v.w
}
```

How close do you get to CUB? What would you need to add if `n` were not a multiple of 4?

Hint: on our RTX 4090, this one change took `v2` from 484 GB/s to about 900 GB/s, within a few percent of `cub::DeviceHistogram`.

In summary, a scan keeps every partial sum of a reduction. Inside a block, we scan with a Hillis-Steele loop in shared memory, or faster, with `__shfl_up_sync` inside each warp. For a large array, we scan each block, scan the block totals, and add them back, which costs twice the traffic of a copy; CUB avoids that with a single pass. For histograms, many threads updating a few counters is the enemy: giving each block a private copy in shared memory and merging once at the end made our histogram 100 times faster. In the next post, we will look at CCCL (Thrust, CUB, and libcu++) and the other CUDA libraries, so we don't have to write these patterns by hand.

{{< quiz >}}
- q: "What is the inclusive scan of `3 1 4 1`?"
  options: ["`0 3 4 8`", "`3 4 8 9`", "`9 9 9 9`", "`3 1 4 1`"]
  answer: 1
  explain: "Each output is the sum of its own element and everything before it. `0 3 4 8` is the *exclusive* scan."
- q: "Why is the histogram with global atomics so slow?"
  options:
    - "`atomicAdd` does not work on global memory."
    - "Millions of atomics go to only 256 addresses, and atomics on the same address run one after another."
    - "The grid-stride loop reads the input twice."
  answer: 1
  explain: "The work is fine; the contention is the problem. Privatization gives each block its own bins in shared memory, so only a few global atomics are left."
- q: "Each block has scanned its own piece of a large array. What must still be added to every element of a block?"
  options:
    - "The total of the block itself."
    - "The sum of the totals of all earlier blocks."
    - "Nothing, the result is already correct."
  answer: 1
  explain: "That is why we scan the block totals and add them back: the scanned total of the earlier blocks is each block's offset."
{{< /quiz >}}

[1]: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back "Merrill and Garland: Single-pass Parallel Prefix Scan with Decoupled Look-back"
[2]: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda "GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA"
[3]: https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/ "GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell"
[4]: https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html "CUB DeviceScan"
[5]: https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceHistogram.html "CUB DeviceHistogram"
