---
title: "Post 09: Reduction: Summing 64 Million Numbers"
date: 2026-09-29T19:09:00+03:00
weight: 10
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

So far, every kernel in this series has been "one thread, one output": thread `i` reads element `i` and writes element `i`. The threads never needed to talk to each other. In this post, we start the patterns stage with the simplest problem where they must: adding up an array into a single number. This is called a *reduction*, and the same idea gives you max, min, dot products, and norms.

On the CPU, it is one loop. On the GPU, we have millions of threads and only one result, so somebody has to combine all those partial sums. We will write four versions, each one fixing the bottleneck of the one before, then let CUB do it in one call. As in post 03, every version gets a score in GB/s.

## What We Will Learn

In this article, we will cover:

* Why a sum is harder on the GPU than a vector add.
* Global atomics, and what the compiler secretly does with them.
* A tree reduction in shared memory.
* Warp shuffles with `__shfl_down_sync`.
* Cooperative groups: `cg::tiled_partition` and `cg::reduce`.
* `cub::DeviceReduce::Sum` and its two-step temporary storage call.
* Why summing floats gives different answers depending on the order.

## Table of Contents

1. The Full Program.
2. The Input and the Score.
3. v1: Global Atomics.
4. v2: Shared-Memory Tree.
5. v3: Warp Shuffle.
6. v4: Cooperative Groups.
7. CUB: One Call.
8. Build and Run.
9. Exercise.

---

## 1. The Full Program

```bash
mkdir reduction
cd reduction
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post09/main.cu" >}}

It is longer than usual because it holds five reductions side by side. We will go through them one at a time.

---

## 2. The Input and the Score

{{< code file="cuda/post09/main.cu" region="host" link="false" >}}

We sum 2^26 = 67,108,864 numbers. That is 268 MB, much bigger than the 72 MB L2 cache of the RTX 4090, so every run really reads from VRAM.

We use `int` on purpose. Integer addition is *associative*: `(a + b) + c` is exactly `a + (b + c)`. Our four versions add the numbers in four different orders, and with `int` they must all give exactly `100663296`. With `float` that is not true, and we will come back to it in the exercise.

A reduction reads every input once and writes almost nothing, so it is memory-bound. Its score is the number of bytes read divided by the time:

{{< code file="cuda/post09/main.cu" region="report" link="false" >}}

We use the `time_ms` helper from post 03 (3 warm-up runs, then the average of 20). Each version writes into a single `int` on the device with `atomicAdd`, so every timed run first zeroes it with `cudaMemsetAsync`. The memset writes 4 bytes, so its cost is tiny next to reading 268 MB.

To know what "fast" means, we also time a device-to-device copy of the same buffer, like in post 03:

{{< code file="cuda/post09/main.cu" region="ceiling" link="false" >}}

A copy reads and writes every byte, so it moves `2 * bytes`. On our machine it reaches about 920 GB/s. That is the practical ceiling. No kernel that has to read 268 MB will beat it by much.

We launch 256 threads per block (`kThreads`) for every hand-written version.

---

## 3. v1: Global Atomics

{{< code file="cuda/post09/main.cu" region="atomic" link="false" >}}

This is the most direct translation of the CPU loop. Every thread adds its element to `*out`.

We cannot write `*out += in[i]`. That line is really three steps: read `*out`, add, write it back. If two threads do that at the same time, both read the same old value and one of the additions is lost. `atomicAdd` does the read, add, and write as one step that no other thread can interrupt. The hardware handles it in the L2 cache.

The problem is that all 67 million additions go to the same address, so they cannot happen in parallel. They wait in line.

Output:

```text
v1 global atomics:       0.970 ms,  277 GB/s (27.4% of peak), sum = 100663296 OK
```

277 GB/s is slow, but it is much faster than 67 million atomics in a row should be. The reason is that the compiler rewrote our kernel. We can see the machine code (called SASS) with `cuobjdump`, which comes with the toolkit:

```bash
cuobjdump -sass -fun _Z13reduce_atomicPKiPii ./reduction | grep -E "VOTEU|REDUX|RED\.E"
```

```text
        /*00a0*/                   VOTEU.ANY UR4, UPT, PT ;                      /* 0x0000000000047886 */
        /*0100*/                   REDUX.SUM UR5, R2 ;                           /* 0x00000000020573c4 */
        /*0120*/               @P0 RED.E.ADD.STRONG.GPU [R4.64], R7 ;            /* 0x000000070400098e */
```

`_Z13reduce_atomicPKiPii` is the C++ mangled name of `reduce_atomic`. The compiler noticed that all 32 threads of a warp add to the same address. So it first sums the 32 values inside the warp (`REDUX.SUM`, a hardware instruction on compute capability 8.0 and newer), then only one thread does the global atomic (`RED.E.ADD`). This is called a *warp-aggregated atomic*. We send one atomic per warp, 2 million in total, instead of 67 million.

The compiler can only do this for integers, because it changes the order of the additions. In the exercise, you will see how slow v1 becomes with `float`, when the compiler is not allowed to help.

---

## 4. v2: Shared-Memory Tree

The fix is to add numbers locally first and only send a few partial sums to global memory. Inside a block, threads can share data through shared memory, which we met in post 06.

{{< code file="cuda/post09/main.cu" region="shared" link="false" >}}

Each thread loads one element into `s`. Threads past the end load `0`, which does not change a sum. Then we add pairs, halving the number of active threads each step:

```text
step 1 (stride 128): s[0] += s[128], s[1] += s[129], ..., s[127] += s[255]
step 2 (stride  64): s[0] += s[64],  ...,                s[63]  += s[127]
...
step 8 (stride   1): s[0] += s[1]
```

After 8 steps, `s[0]` holds the sum of the block's 256 elements, and thread 0 adds it to `*out`. That is one global atomic per block: 262,144 instead of 2 million.

The `__syncthreads()` inside the loop is required. Step 2 reads `s[64]`, which another thread wrote in step 1. Without the barrier, it may read it before the write happens. That is the kind of race we caught with `compute-sanitizer --tool racecheck` in post 06.

Notice that active threads are `tid < stride`, a contiguous range. Warps are either fully active or fully idle until the stride drops below 32, so there is little branch divergence. Contiguous `s[tid]` and `s[tid + stride]` also avoid shared-memory bank conflicts.

Output:

```text
v2 shared tree:          0.378 ms,  711 GB/s (70.5% of peak), sum = 100663296 OK
```

2.6 times faster than v1, but still under the copy ceiling. Each block reads only 1 KB from memory and then spends 8 steps and 9 barriers on it. During those steps, half the threads, then three quarters, then more, sit idle, and the whole block waits at every barrier.

---

## 5. v3: Warp Shuffle

The 32 threads of a warp run together, so they do not need shared memory or `__syncthreads` to exchange values. A *shuffle* lets a thread read a register of another thread in the same warp directly:

{{< code file="cuda/post09/main.cu" region="shuffle" link="false" >}}

`__shfl_down_sync(mask, v, offset)` returns the value of `v` from the lane `offset` places higher. Lane 0 gets lane 16's value, lane 1 gets lane 17's, and so on. The `0xffffffff` mask says all 32 lanes take part. After 5 steps (offsets 16, 8, 4, 2, 1), lane 0 holds the sum of the whole warp. The other lanes hold partial sums we simply ignore.

A block of 256 threads has 8 warps. Each warp reduces its 32 values with shuffles, and lane 0 writes the result to `warp_sums`. After one `__syncthreads()`, the first warp loads those 8 values and reduces them with the same `warp_sum`. Lanes 8 to 31 load `0`. So we have one barrier per block instead of nine.

Output:

```text
v3 warp shuffle:         0.282 ms,  950 GB/s (94.3% of peak), sum = 100663296 OK
```

950 GB/s is a bit *above* our copy ceiling of 920 GB/s. A copy mixes reads and writes, and DRAM has to switch between the two. A reduction only reads, which is slightly easier for the memory. Either way, v3 is at the limit of the hardware. From here on, we cannot go faster, only simpler.

---

## 6. v4: Cooperative Groups

v3 works, but the code is full of details: the mask, the magic 16, `% 32`, `/ 32`. Cooperative groups is a small C++ library that ships with CUDA and gives names to these groups of threads:

{{< code file="cuda/post09/main.cu" region="cg" link="false" >}}

`cg::this_thread_block()` is the block, and `cg::tiled_partition<32>(block)` splits it into groups of 32 threads, which are the warps. `cg::reduce(warp, v, cg::plus<int>())` sums `v` across the tile and gives the result to *every* thread in it, not only lane 0. `warp.thread_rank()` is the lane. We include `<cooperative_groups.h>` and `<cooperative_groups/reduce.h>`, and use the short alias `namespace cg = cooperative_groups`.

We also change the launch. Instead of one element per thread, we start just enough blocks to fill the GPU, and each thread walks through the array with the grid-stride loop from post 02:

{{< code file="cuda/post09/main.cu" region="cg-launch" link="false" >}}

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (post 08) tells us how many of these blocks fit on one SM at the same time. On the RTX 4090, that is 6 blocks on each of 128 SMs, so 768 blocks. Each thread now adds about 340 numbers in a register before it talks to anyone. Only one atomic per warp goes to global memory, 6,144 in total. We do not even need shared memory anymore.

Output:

```text
v4 cooperative groups:   0.285 ms,  941 GB/s (93.3% of peak), sum = 100663296 OK
```

The same speed as v3, within 1%. That is expected: v3 already hit the memory limit. What we gained is shorter code and 40 times fewer atomics, which matters when the input is small or the GPU is busy with other work. `cg::reduce` is not slower than hand-written shuffles either. For `int` on our GPU, it compiles to the same `REDUX.SUM` instruction we saw in v1. On older GPUs (Turing), it falls back to shuffles.

---

## 7. CUB: One Call

We wrote four kernels to learn how a reduction works. In real code, we would use CUB, the library of GPU building blocks that is part of CCCL and ships with the toolkit:

{{< code file="cuda/post09/main.cu" region="cub" link="false" >}}

CUB needs some scratch memory for partial sums, and it does not allocate it by itself. So every CUB device algorithm is called twice:

1. With `nullptr` as the temporary storage. CUB only writes the required size into `temp_bytes` and returns. Nothing runs on the GPU.
2. With a real buffer of that size. This call does the work.

On our machine, CUB asks for 15,615 bytes. We allocate it with the same `make_device_buffer`, now a template so it can hold `char` instead of `float`. CUB writes the result into `*out` directly, so there is no memset in this lambda.

Output:

```text
cub::DeviceReduce::Sum:  0.285 ms,  943 GB/s (93.6% of peak), sum = 100663296 OK
```

The same speed as our best hand-written kernel, and it handles any type, any size, and any GPU, with tuning for each architecture.

One more thing about the includes. CUB pulls in the C++ `<mutex>` header, which uses two glibc functions that are only declared when `_GNU_SOURCE` is defined. Our `-U_GNU_SOURCE` workaround from post 00 hides them, and the build fails with `identifier "pthread_cond_clockwait" is undefined`. The fix is to declare them ourselves before any CUDA include:

{{< code file="cuda/post09/main.cu" region="shim" link="false" >}}

The functions exist in glibc, only their declarations were hidden, so this links fine. Like `-U_GNU_SOURCE` itself, drop it once a newer toolkit fixes the header clash.

---

## 8. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o reduction
./reduction
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post09/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/reduction
```
{{< /tab >}}
{{< /tabs >}}

We do not link anything extra: CUB and cooperative groups are header-only and come with the toolkit.

Output: (RTX 4090)

```text
n = 67108864 ints (268 MB), expected sum = 100663296

copy (ceiling):          0.583 ms,  921 GB/s (91.4% of peak)
v1 global atomics:       0.970 ms,  277 GB/s (27.4% of peak), sum = 100663296 OK
v2 shared tree:          0.378 ms,  711 GB/s (70.5% of peak), sum = 100663296 OK
v3 warp shuffle:         0.282 ms,  950 GB/s (94.3% of peak), sum = 100663296 OK
v4 cooperative groups:   0.285 ms,  941 GB/s (93.3% of peak), sum = 100663296 OK
cub::DeviceReduce::Sum:  0.285 ms,  943 GB/s (93.6% of peak), sum = 100663296 OK

v4 grid: 128 SMs x 6 blocks = 768 blocks
CUB temp storage: 15615 bytes
PASSED
```

| Version               | Time (ms) | GB/s | vs. copy ceiling (921 GB/s) | Global atomics |
| --------------------- | --------- | ---- | --------------------------- | -------------- |
| v1 global atomics     | 0.970     | 277  | 30%                         | 2,097,152 (one per warp, thanks to the compiler) |
| v2 shared-memory tree | 0.378     | 711  | 77%                         | 262,144 (one per block) |
| v3 warp shuffle       | 0.282     | 950  | 103%                        | 262,144 (one per block) |
| v4 cooperative groups | 0.285     | 941  | 102%                        | 6,144 (one per warp) |
| CUB                   | 0.285     | 943  | 102%                        | none           |

We ran the program three times. In the first run, v1 and v2 were about 7% slower (258 and 661 GB/s); the other two runs matched each other within 1%.

---

## 9. Exercise

Switch the program to `float`. Change `int` to `float` in the kernels, the buffers, `warp_sum`, and `cg::plus`, and fill the input with `h_in[i] = 0.1f * (i % 4)`. Compute the expected value on the CPU in two ways: once with a `double` accumulator and once with a `float` one. Print the sums with `%.1f`.

Before you run it, guess: will all versions agree? Here is what we got:

```text
expected (double):  10066329.9
CPU float loop:      8388608.0
v1 global atomics:   8388608.0   (87.4 ms, 3 GB/s)
v2 shared tree:     10064070.0
v3 warp shuffle:    10064070.0
v4 cooperative:     10066746.0   (10066747.0 on the next run)
CUB:                10066329.0
```

A few things to explain:

* **The CPU loop is the worst.** A `float` has 24 bits of precision. Once the running total reaches 2^23 = 8,388,608, adding 0.3 rounds back to the same number, and the sum stops growing. v1 adds one number at a time too, so it gets stuck at the same value.
* **v1 is 90 times slower than with `int`.** Float addition is not associative, so the compiler is not allowed to reorder the adds into a warp-aggregated atomic. Now every one of the 67 million atomics really goes to L2 one after another. This is the real cost of the naive version.
* **The tree versions are more accurate.** They add small partial sums of similar size, so less precision is lost at each step.
* **v4 changes between runs.** The per-warp atomics arrive in a different order each time, and with floats, a different order gives a slightly different result.

So, for floats, never test a reduction with `==`. Compare against a `double` reference with a relative tolerance, for example `std::abs(sum - expected) <= 1e-3 * std::abs(expected)`. Pick the tolerance from the numbers, not from a habit: v2 is off by about 2e-4 here, so a tolerance of 1e-4 would wrongly fail it. Also expect the last digits to move from run to run when atomics are involved. If you need the exact same bits every time, use a fixed order with no atomics, the way CUB does: it gave the same result in every run.

In summary, a reduction is the first pattern where threads must cooperate. Atomics alone serialize on one address. A shared-memory tree adds locally but spends most of its time at barriers. Warp shuffles remove the barriers and reach the memory limit, and cooperative groups give the same speed with clearer code and far fewer atomics. In practice, use `cub::DeviceReduce::Sum`: two calls, one to ask for the temporary storage size and one to do the work, and it runs as fast as our best kernel. With integers, every version gives the same answer. With floats, the order of additions changes the result, so check it with a tolerance.

[1]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf "Mark Harris: Optimizing Parallel Reduction in CUDA"
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions "CUDA Programming Guide: Warp Shuffle Functions"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups "CUDA Programming Guide: Cooperative Groups"
[4]: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html "CUB: DeviceReduce"
[5]: https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/ "CUDA Pro Tip: Warp-Aggregated Atomics"
