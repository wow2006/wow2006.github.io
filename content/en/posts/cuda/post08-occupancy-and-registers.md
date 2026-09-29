---
title: "Post 08: Occupancy and registers"
date: 2026-09-29T19:08:00+03:00
weight: 9
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the previous post, we built a tiled matrix multiply. Every block asked for a tile of shared memory, and every thread kept a running sum in a register. Both of these come from a fixed budget on each SM. The more a block asks for, the fewer blocks fit on the SM at the same time. How full an SM is, measured in warps, is called *occupancy*.

You will often read that high occupancy is the goal. In this post, we will see what occupancy is, what limits it, and how to measure it. Then we will run an experiment that shows a kernel at 17% occupancy running exactly as fast as at 100%, and a kernel that gets 1.7 times *slower* when we force its occupancy up.

## What We Will Learn

In this article, we will cover:

* What occupancy is, and the four resources that limit it.
* How to see how many registers a kernel uses with `nvcc -Xptxas -v`.
* How to ask the runtime for occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and `cudaOccupancyMaxPotentialBlockSize`.
* Why higher occupancy is not always faster, with measured numbers.
* How `__launch_bounds__` changes register use, and what it can cost.
* How to read the Occupancy and Launch Statistics sections of `ncu`.

## Table of Contents

1. The Full Program.
2. What Occupancy Is.
3. Registers per Thread.
4. How Many Blocks Fit on an SM.
5. Asking the Runtime.
6. Build and Run.
7. Higher Occupancy Is Not Faster.
8. `__launch_bounds__` and Forced Occupancy.
9. Occupancy in Nsight Compute.
10. Back to the Blur from Post 02.
11. Exercise.

---

## 1. The Full Program

```bash
mkdir occupancy
cd occupancy
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post08/main.cu" >}}

We keep `CUDA_CHECK`, `DeviceBuffer`, and the `time_ms` helper from post 03. The kernel is the simplest memory-bound kernel there is: `y = a * x` over 64 million floats. That is on purpose. It does almost no math, so its speed depends only on how well it keeps the memory system busy, and that is exactly what occupancy is about.

---

## 2. What Occupancy Is

An SM does not run one block at a time. It keeps several blocks *resident*, and it switches between their warps every clock cycle. When one warp waits for memory, the SM issues an instruction from another warp that is ready. This is how a GPU hides the long wait for DRAM: not with big caches, but with many warps to choose from.

Occupancy is the number of resident warps on an SM divided by the maximum it can hold:

```text
occupancy = resident warps per SM / max warps per SM
```

The maximum depends on the GPU, so we ask `cudaGetDeviceProperties` instead of hard-coding it:

{{< code file="cuda/post08/main.cu" region="props" link="false" >}}

On the RTX 4090, this prints:

```text
NVIDIA GeForce RTX 4090 (sm_89), 128 SMs
  max threads per SM      : 1536 (48 warps)
  max blocks per SM       : 24
  registers per SM        : 65536
  shared memory per SM    : 102400 bytes
  shared memory per block : 101376 bytes (opt-in max)
  reserved smem per block : 1024 bytes
```

Every SM can hold at most 48 warps (1536 threads) and at most 24 blocks. All resident threads share 65,536 registers and 100 KB of shared memory. The driver also reserves 1 KB of shared memory for each block for its own use. These numbers change between architectures; an A100 holds 64 warps per SM, for example. That is why we query them.

---

## 3. Registers per Thread

Registers are the fastest memory on the GPU. Every local variable in a kernel lives in a register, if the compiler can manage it. The compiler decides how many registers each thread needs, and we can ask it to tell us with `-Xptxas -v` (`-Xptxas` passes an option to `ptxas`, the part of `nvcc` that turns PTX into machine code):

```bash
nvcc -U_GNU_SOURCE -arch=native -O3 -Xptxas -v main.cu -o occupancy
```

`-arch=native` compiles for the GPU in this machine, like `CMAKE_CUDA_ARCHITECTURES native` in CMake. Without it, `nvcc` compiles for an older default architecture, and the register counts it reports are for that one, not for our sm_89.

```text
ptxas info    : Compiling entry function '_Z5scaleILi32ELi6EEvPKfPffi' for 'sm_89'
ptxas info    : Function properties for _Z5scaleILi32ELi6EEvPKfPffi
    120 bytes stack frame, 116 bytes spill stores, 116 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 120 bytes cumulative stack size, 376 bytes cmem[0]
ptxas info    : Compiling entry function '_Z5scaleILi4ELi1EEvPKfPffi' for 'sm_89'
ptxas info    : Function properties for _Z5scaleILi4ELi1EEvPKfPffi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compiling entry function '_Z5scaleILi32ELi1EEvPKfPffi' for 'sm_89'
ptxas info    : Function properties for _Z5scaleILi32ELi1EEvPKfPffi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compiling entry function '_Z5scaleILi1ELi1EEvPKfPffi' for 'sm_89'
ptxas info    : Function properties for _Z5scaleILi1ELi1EEvPKfPffi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 376 bytes cmem[0]
```

The names are mangled C++, but you can read the template arguments: `scaleILi4ELi1E` is `scale<4, 1>`. For each kernel we get two useful lines:

* **Used N registers**: registers per thread.
* **spill stores / spill loads**: when a kernel needs more registers than it is allowed, the compiler moves some values out to *local memory*. Despite the name, local memory is private to each thread but lives in the same DRAM as global memory, cached in L1 and L2. Zero spills is what we want. One of our kernels spills 116 bytes per thread; we will come back to it in section 8.

Now look at the kernel:

{{< code file="cuda/post08/main.cu" region="kernel" link="false" >}}

`ILP` stands for *instruction-level parallelism*. With `ILP = 1`, each thread loads one float, scales it, and stores it. With `ILP = 4`, it first issues 4 independent loads, then does the 4 stores. The loads do not depend on each other, so the thread does not wait for the first one before sending the second. The elements are `blockDim.x` apart, so the 32 threads of a warp still read 32 neighbouring floats in every load, as we learned in post 05.

The array `v[ILP]` is indexed only with constants after `#pragma unroll`, so the compiler keeps it in registers. That is why the register count grows with `ILP`: 8 registers for `scale<1>`, 22 for `scale<4>`, and 80 for `scale<32>`.

We can also read the register count at runtime with `cudaFuncGetAttributes`, which is what our program does in its `regs=` column.

---

## 4. How Many Blocks Fit on an SM

A block is placed on an SM only if there is room for *all* of it: all its warps, all its registers, and all its shared memory. So the number of resident blocks per SM is the smallest of four limits:

| Limit                | RTX 4090 budget | Blocks that fit                                   |
| -------------------- | --------------- | ------------------------------------------------- |
| Threads (warps)      | 1536 threads    | 1536 / threads per block                          |
| Blocks               | 24 blocks       | 24                                                |
| Registers            | 65,536          | 65,536 / (registers per thread × threads per block) |
| Shared memory        | 102,400 bytes   | 102,400 / (shared memory per block + 1 KB)        |

Registers are handed out per warp in chunks of 256, which means a thread's count is rounded up to a multiple of 8. Let's do the math for our kernels with 256 threads per block:

* `scale<1>`: 8 registers → 2,048 per block → 32 blocks by registers. The thread limit gives 1536 / 256 = 6 blocks, so 6 blocks, 48 warps, **100%**.
* `scale<4>`: 22 registers, rounded to 24 → 6,144 per block → 10 blocks by registers. Still 6 blocks by threads, **100%**.
* `scale<32>`: 80 registers → 20,480 per block → 65,536 / 20,480 = 3.2, so **3 blocks**. Registers are now the limit: 24 warps, **50%**.

Shared memory works the same way. Our kernel does not use shared memory at all, but a launch can still *reserve* it with the third launch parameter, `<<<blocks, threads, smem>>>`. The SM has to set that much aside for each block, used or not. We will use this trick to lower occupancy on purpose. For example, 50,176 bytes per block plus the 1 KB reserve is 51,200 bytes, and only 2 of those fit in 102,400 bytes: 16 warps, 33%.

---

## 5. Asking the Runtime

Doing this math by hand is good for understanding, but the runtime can do it for us, and it knows all the rounding rules. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` takes a kernel, a block size, and the dynamic shared memory per block, and returns how many blocks fit on one SM:

{{< code file="cuda/post08/main.cu" region="run" link="false" >}}

A few details:

* By default a block may use at most 48 KB of dynamic shared memory. To ask for more (up to the 101,376 bytes opt-in maximum we printed), we must first raise the limit with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)`.
* We compute occupancy as `blocksPerSM × threads / 32` warps, divided by 48.
* Bandwidth counts one read and one write per element, and we compare it with the 1008 GB/s peak we derived in post 03.

The second API, `cudaOccupancyMaxPotentialBlockSize`, answers a different question: *which block size gives the highest occupancy for this kernel?*

{{< code file="cuda/post08/main.cu" region="potential" link="false" >}}

```text
MaxPotentialBlockSize(scale<1>) : block=256, min grid=768
MaxPotentialBlockSize(scale<32>): block=256, min grid=384
```

`block` is the suggested block size. It cannot be more than 256 here, because `__launch_bounds__(256, ...)` on our kernel promises the compiler that we never launch more than 256 threads per block (section 8). `min grid` is the number of blocks that fills the whole GPU once: 6 blocks × 128 SMs = 768 for `scale<1>`, and only 3 × 128 = 384 for `scale<32>`, because of its registers.

This API is a useful starting point when you have no better idea. But notice what it optimizes: occupancy, not speed. As we will see next, those are not the same thing.

---

## 6. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 -Xptxas -v main.cu -o occupancy
./occupancy
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post08/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/occupancy
```
{{< /tab >}}
{{< /tabs >}}

Output: (RTX 4090)

```text
NVIDIA GeForce RTX 4090 (sm_89), 128 SMs
  max threads per SM      : 1536 (48 warps)
  max blocks per SM       : 24
  registers per SM        : 65536
  shared memory per SM    : 102400 bytes
  shared memory per block : 101376 bytes (opt-in max)
  reserved smem per block : 1024 bytes

MaxPotentialBlockSize(scale<1>) : block=256, min grid=768
MaxPotentialBlockSize(scale<32>): block=256, min grid=384

scale< 1,1> regs= 8 smem=     0  6 blocks/SM  48/48 warps (100.0%)  0.583 ms, 921 GB/s (91.4% of peak)
scale< 4,1> regs=22 smem=     0  6 blocks/SM  48/48 warps (100.0%)  0.584 ms, 919 GB/s (91.2% of peak)
scale< 1,1> regs= 8 smem= 19456  5 blocks/SM  40/48 warps ( 83.3%)  0.582 ms, 922 GB/s (91.5% of peak)
scale< 4,1> regs=22 smem= 19456  5 blocks/SM  40/48 warps ( 83.3%)  0.584 ms, 919 GB/s (91.1% of peak)
scale< 1,1> regs= 8 smem= 24576  4 blocks/SM  32/48 warps ( 66.7%)  0.582 ms, 923 GB/s (91.5% of peak)
scale< 4,1> regs=22 smem= 24576  4 blocks/SM  32/48 warps ( 66.7%)  0.584 ms, 920 GB/s (91.2% of peak)
scale< 1,1> regs= 8 smem= 32768  3 blocks/SM  24/48 warps ( 50.0%)  0.588 ms, 913 GB/s (90.6% of peak)
scale< 4,1> regs=22 smem= 32768  3 blocks/SM  24/48 warps ( 50.0%)  0.583 ms, 921 GB/s (91.3% of peak)
scale< 1,1> regs= 8 smem= 50176  2 blocks/SM  16/48 warps ( 33.3%)  0.650 ms, 826 GB/s (82.0% of peak)
scale< 4,1> regs=22 smem= 50176  2 blocks/SM  16/48 warps ( 33.3%)  0.582 ms, 923 GB/s (91.5% of peak)
scale< 1,1> regs= 8 smem=101376  1 blocks/SM   8/48 warps ( 16.7%)  1.019 ms, 527 GB/s (52.2% of peak)
scale< 4,1> regs=22 smem=101376  1 blocks/SM   8/48 warps ( 16.7%)  0.581 ms, 925 GB/s (91.7% of peak)

scale<32,1> regs=80 smem=     0  3 blocks/SM  24/48 warps ( 50.0%)  0.594 ms, 904 GB/s (89.6% of peak)
scale<32,6> regs=40 smem=     0  6 blocks/SM  48/48 warps (100.0%)  0.987 ms, 544 GB/s (54.0% of peak)
PASSED (0 errors)
```

Run it a few times. On our machine, the numbers move by about 1 to 2% between runs, so treat differences smaller than that as noise.

---

## 7. Higher Occupancy Is Not Faster

This is the loop that produced the middle part of the output:

{{< code file="cuda/post08/main.cu" region="sweep" link="false" >}}

For each target of 6, 5, ..., 1 blocks per SM, we give each block an equal slice of the SM's shared memory, minus the 1 KB reserve, rounded down to a whole KB. The kernel never touches it. It only exists to stop more blocks from fitting. The runtime confirms we got exactly the occupancy we asked for, from 100% down to 16.7%.

Here are the results side by side:

| Warps per SM | Occupancy | `scale<1>`         | `scale<4>`         |
| ------------ | --------- | ------------------ | ------------------ |
| 48           | 100%      | 0.583 ms, 921 GB/s | 0.584 ms, 919 GB/s |
| 40           | 83%       | 0.582 ms, 922 GB/s | 0.584 ms, 919 GB/s |
| 32           | 67%       | 0.582 ms, 923 GB/s | 0.584 ms, 920 GB/s |
| 24           | 50%       | 0.588 ms, 913 GB/s | 0.583 ms, 921 GB/s |
| 16           | 33%       | 0.650 ms, 826 GB/s | 0.582 ms, 923 GB/s |
| 8            | 17%       | 1.019 ms, 527 GB/s | 0.581 ms, 925 GB/s |

Two things stand out.

First, `scale<1>` does not care about occupancy until it drops below 50%. Going from 100% to 50% changed nothing. Only at 33% and 17% does it slow down, and at 17% it runs at half speed.

Second, `scale<4>` does not slow down *at all*. At 17% occupancy, with only 8 warps per SM, it moves 925 GB/s, the same as at 100%. The GPU is equally busy with one sixth of the warps.

Why? Remember why we wanted many warps: to have enough memory requests waiting at the same time. DRAM takes a long time to answer, so to keep ~1 TB/s flowing, a lot of bytes must be *in flight* at any moment. Warps are one way to get requests in flight. Independent loads inside a thread are another. Count the bytes each SM can have requested before any thread must stop and wait:

| Kernel     | Warps per SM | Loads in flight per thread | Bytes in flight per SM | Bandwidth |
| ---------- | ------------ | -------------------------- | ---------------------- | --------- |
| `scale<1>` | 48           | 1                          | 6 KB                   | 921 GB/s  |
| `scale<1>` | 24           | 1                          | 3 KB                   | 913 GB/s  |
| `scale<1>` | 16           | 1                          | 2 KB                   | 826 GB/s  |
| `scale<1>` | 8            | 1                          | 1 KB                   | 527 GB/s  |
| `scale<4>` | 8            | 4                          | 4 KB                   | 925 GB/s  |

(Bytes in flight = warps × 32 threads × loads per thread × 4 bytes.)

The pattern is clear: on this GPU, about 3 KB in flight per SM is enough to saturate memory. It does not matter whether that comes from 24 warps with one load each or from 8 warps with four loads each. Occupancy is only a means to an end. What the hardware needs is enough independent work, and occupancy is just one way to supply it. This idea is from Vasily Volkov's well-known talk *Better Performance at Lower Occupancy* [1].

Why would you ever *want* lower occupancy? Because the resources that lower it are useful. A tiled matrix multiply wants big shared-memory tiles and many registers per thread to reuse data. Both lower occupancy, and both make the kernel faster. Fast GEMM kernels often run at low occupancy on purpose; we will see this in post 18.

---

## 8. `__launch_bounds__` and Forced Occupancy

The last two lines of the output test the opposite direction:

{{< code file="cuda/post08/main.cu" region="registers" link="false" >}}

`scale<32>` issues 32 loads per thread before its first store. It needs 80 registers to hold them, and as we computed in section 4, 80 registers limit it to 3 blocks per SM, 50% occupancy. Still, it runs at 904 GB/s, nearly full speed.

Now suppose someone looks at the ncu report, sees "50% occupancy", and decides to fix it. The tool for that is `__launch_bounds__`, which we put on our kernel:

```cpp
template <int ILP, int MinBlocks = 1>
__global__ void __launch_bounds__(256, MinBlocks)
scale(const float* x, float* y, float a, int n)
```

It has two arguments:

* **Max threads per block** (256): a promise that we never launch this kernel with more than 256 threads per block. The compiler can plan its register use for that block size, and the runtime refuses a launch with more threads than promised.
* **Min blocks per SM** (optional): a request that at least this many blocks fit on one SM. The compiler caps registers to make that possible.

For `scale<32, 6>`, the compiler must fit 6 blocks × 256 threads = 1536 threads into 65,536 registers. That is 42.7 registers per thread, rounded down to a multiple of 8: **40**. The `-Xptxas -v` output from section 3 shows the result: `Used 40 registers`, but also `116 bytes spill stores, 116 bytes spill loads`. The 32 values do not fit in 40 registers, so the compiler parks some of them in local memory.

The runtime reports what we asked for: 6 blocks per SM, 48 warps, 100% occupancy. And the kernel is **1.7 times slower**: 0.987 ms instead of 0.594 ms, 544 GB/s instead of 904 GB/s. Every spilled value is an extra store and an extra load through the caches, in a kernel whose whole job was to keep memory busy with useful traffic.

So `__launch_bounds__` is a real tool, but use it to *remove* a limit you measured, not to chase a number. Some good uses:

* Always give the max threads per block when you know it. It costs nothing, and it lets the compiler use more registers when the block is small.
* Try a min blocks value when `-Xptxas -v` shows a kernel just above a register boundary, for example 66 registers where 64 would let one more block fit. Then measure. If ptxas starts reporting spills, it is probably not worth it.

`nvcc -maxrregcount=N` does the same thing for every kernel in a file. `__launch_bounds__` is better because it is per kernel.

---

## 9. Occupancy in Nsight Compute

In post 04 we met `ncu`, which reports occupancy for any kernel. For our program, the two relevant sections are `Occupancy` and `LaunchStats`. Each `run` call launches the kernel 23 times (3 warm-up + 20 timed), so `--launch-skip` picks a configuration. This profiles the first launch of `scale<1, 1>` and the first launch of `scale<32, 6>`, the last configuration (13 × 23 = 299 launches in):

```bash
ncu --section Occupancy --section LaunchStats -c 1 ./occupancy
ncu --section Occupancy --section LaunchStats --launch-skip 299 -c 1 ./occupancy
```

On a desktop Linux machine, you will likely get `ERR_NVGPUCTRPERM` at first. As we covered in post 04, either run it with `sudo /usr/local/cuda/bin/ncu ...`, or allow profiling for normal users with the driver option `NVreg_RestrictProfilingToAdminUsers=0` and a reboot. On our test machine, profiling was restricted to admins, so we cannot paste the report here. Here is what to look for when you run it:

**Launch Statistics** shows what you asked for:

* `Block Size`, `Grid Size`: the launch configuration.
* `Registers Per Thread`: should match `-Xptxas -v` (8 and 40 for these two kernels).
* `Dynamic Shared Memory Per Block`, `Static Shared Memory Per Block`: our `smem` argument, and any `__shared__` arrays declared in the kernel.
* `Waves Per SM`: how many rounds of "fill every SM" the grid needs. With 64 million floats and 256 threads, `scale<1>` launches 262,144 blocks, so the waves count is large and the last partial wave does not matter. For small grids, a wave count like 1.1 means the last 0.1 wave runs on a mostly empty GPU.

**Occupancy** shows what that allows:

* `Block Limit Registers`, `Block Limit Shared Mem`, `Block Limit Warps`, `Block Limit SM`: the four limits from our table in section 4. The smallest one decides the occupancy. For `scale<32, 1>` you would see the register limit at 3; for the shared-memory sweep, the shared memory limit.
* `Theoretical Occupancy` and `Theoretical Active Warps per SM`: the result of that calculation. These should match the percentages our program printed, since `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does the same math.
* `Achieved Occupancy` and `Achieved Active Warps Per SM`: what the hardware actually measured, averaged over the run. It is lower than theoretical when blocks finish unevenly, when the grid is too small, or during the last wave.

`ncu` will also print hints such as "theoretical occupancy is limited by the number of required registers". Read them as facts about the kernel, not as instructions. For `scale<32, 1>` that hint is correct, and following it made the kernel 1.7 times slower.

---

## 10. Back to the Blur from Post 02

In post 02, we tried different block shapes for our 5 × 5 box blur, and one result had no explanation yet. A 32 × 32 block took 0.181 ms, while a 16 × 16 block took 0.112 ms. Both have warps that read along a row, so coalescing was not the reason.

Now we can do the math. `-Xptxas -v` on the post 02 program reports `blur2d` using 23 registers per thread. The `cudaOccupancyMaxActiveBlocksPerMultiprocessor` numbers for it are:

| Block   | Threads | Blocks per SM | Warps per SM | Occupancy | Time (post 02 program) |
| ------- | ------- | ------------- | ------------ | --------- | ---------------------- |
| 16 × 16 | 256     | 6             | 48           | 100%      | 0.114 ms               |
| 32 × 16 | 512     | 3             | 48           | 100%      | 0.114 ms               |
| 32 × 24 | 768     | 2             | 48           | 100%      | 0.122 ms               |
| 32 × 32 | 1024    | 1             | 32           | 67%       | 0.181 ms               |

A 1024-thread block uses 1024 of the SM's 1536 thread slots. A second one does not fit, so the remaining 512 slots stay empty for the whole kernel: 67% occupancy is the thread limit, not registers or shared memory.

There is a second cost that the table does not show. With one block per SM, a new block can start only when *all* 32 warps of the old one are done. The first warps to finish leave their slots idle while they wait for the slowest ones. With 6 smaller blocks, a finished block is replaced right away while the other 5 keep running.

Unlike `scale<4>`, the blur relies on many warps to hide memory latency, so losing a third of them hurts. Switching to any block size that divides 1536 evenly brings it back to 100% and to roughly the 16 × 16 time. Here, higher occupancy *was* the fix. The lesson of this post is not that occupancy is useless. It is that occupancy is one way to get enough work in flight, and you should check whether it is the missing one before you chase it.

---

## 11. Exercise

Change `threads` from 256 to 32 and run the program again. Before you look at the output, predict the occupancy of the first line (no shared memory). Which of the four limits sets it?

Here are the first two lines on our RTX 4090:

```text
scale< 1,1> regs= 8 smem=     0  24 blocks/SM  24/48 warps ( 50.0%)  1.061 ms, 506 GB/s (50.2% of peak)
scale< 4,1> regs=22 smem=     0  24 blocks/SM  24/48 warps ( 50.0%)  0.582 ms, 923 GB/s (91.5% of peak)
```

A block of 32 threads is a single warp. The thread limit would allow 48 of them, but an SM holds at most 24 blocks, so we get 24 warps: 50%. This is why very small blocks are a bad idea. Notice also that `scale<1>` runs at half speed with 24 warps of 32-thread blocks, while in our main run it ran at full speed with 24 warps of 256-thread blocks. Same occupancy, different speed. Can you think of why? (Hint: count how many blocks the GPU has to launch, and how short each one is.) And, once more, `scale<4>` does not care.

In summary, occupancy is the fraction of an SM's warp slots that are filled, and it is set by the tightest of four limits: threads, blocks, registers, and shared memory per SM. `nvcc -Xptxas -v` shows registers and spills, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does the occupancy math for you, and ncu's Occupancy section shows the same numbers plus what was actually achieved. But occupancy is a means, not the goal. A memory-bound kernel needs enough bytes in flight, and 8 warps with 4 independent loads each did as well as 48 warps. Forcing occupancy up with `__launch_bounds__` made a kernel 1.7 times slower. In the next post, we start on common parallel patterns with a reduction, where these trade-offs show up again.

[1]: https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf "Vasily Volkov: Better Performance at Lower Occupancy (GTC 2010)"
[2]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy "CUDA C++ Best Practices Guide: Occupancy"
[3]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html "CUDA Runtime API: Occupancy"
[4]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds "CUDA C++ Programming Guide: Launch Bounds"
[5]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "Nsight Compute Profiling Guide"
