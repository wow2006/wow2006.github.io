---
title: "Post 18: Capstone: SGEMM from naive to near-cuBLAS"
date: 2026-09-29T19:18:00+03:00
weight: 19
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 07, we wrote a naive matrix multiply and a shared-memory tiled one, and compared both with cuBLAS. The tiled kernel reached about 6,100 GFLOP/s. cuBLAS reached about 54,000. We ended with a promise: add the missing optimizations one at a time, measure each one, and watch the kernel climb.

This is that post. We start from a kernel that reaches about 1% of cuBLAS and finish at about 80%, using only FP32 CUDA cores, plain CUDA C++, and ideas from the earlier posts: coalescing (post 05), shared memory and bank conflicts (post 06), occupancy and registers (post 08), and asynchronous copies (post 12). Each step is a separate kernel in the same program, and each one is checked against cuBLAS.

The order of the steps follows Simon Boehm's well-known article *How to optimize a CUDA matmul kernel for cuBLAS-like performance* [1]. If you enjoy this post, read his too. It goes further, with autotuning and more kernels.

## What We Will Learn

In this article, we will cover:

* Why the way threads are mapped to outputs can change the speed by 8x.
* How register tiling turns a memory-bound kernel into a compute-bound one.
* How to use 128-bit loads (`float4`) in global and shared memory.
* How to copy the next tile with `cp.async` while computing the current one.
* How warp tiling removes shared-memory bank conflicts.
* How to explain every step with a number: instruction counts, registers, and occupancy.

## Table of Contents

1. The Full Program.
2. The Test Bench.
3. Step 1: Naive.
4. Step 2: Coalesced Access.
5. Step 3: Shared-Memory Tiles.
6. Step 4: 1D Register Tiling.
7. Step 5: 2D Register Tiling.
8. Step 6: 128-bit Loads.
9. Step 7: Double Buffering with `cp.async`.
10. Step 8: Warp Tiling.
11. Build and Run.
12. The Scoreboard.
13. Profiling the Steps with Nsight Compute.
14. Exercise.

---

## 1. The Full Program

```bash
mkdir sgemm
cd sgemm
code main.cu
```

You can write the following in `main.cu`. It is long, because it holds eight kernels. We will go through them one by one.

{{< code file="cuda/post18/main.cu" >}}

All kernels compute `C = A · B` for square, row-major `n × n` matrices with `n = 4096`. That is 2 × 4096³ ≈ 137 GFLOP of work per multiply. To keep the code short, the kernels assume `n` is a multiple of 128 and have no bounds checks. Post 07's exercise showed how to handle any size.

---

## 2. The Test Bench

Before we optimize anything, we need a way to measure and a way to check. We reuse both from earlier posts.

The reference is cuBLAS, called with the column-major trick from post 07: passing `B` before `A` gives us a row-major `C`:

{{< code file="cuda/post18/main.cu" region="cublas" link="false" >}}

The timer is the `time_ms` function from post 03: three warm-up runs, then the average of twenty runs measured with `cudaEvent`:

{{< code file="cuda/post18/main.cu" region="time_ms" link="false" >}}

Each kernel needs its own grid and block shape, so a small `switch` holds all the launches in one place:

{{< code file="cuda/post18/main.cu" region="launch" link="false" >}}

Finally, `main` times each kernel, copies `C` back, and compares it with the cuBLAS result. As in post 07, the largest difference must stay below `1e-3`:

{{< code file="cuda/post18/main.cu" region="verify" link="false" >}}

We clear `C` before every kernel, so a kernel that writes nothing cannot pass by reusing the previous result. The program also takes an optional kernel number. `./sgemm 7` runs cuBLAS and only step 7. This makes profiling one kernel with `ncu` easy, as we will see in section 13.

---

## 3. Step 1: Naive

{{< code file="cuda/post18/main.cu" region="naive" link="false" >}}

One thread computes one element of `C`: a dot product of one row of `A` and one column of `B`. This looks just like post 07's naive kernel, with one small difference. Here, `threadIdx.x` picks the *row*, not the column. It is an easy mistake to make, since we usually say "x" first and "row" first.

Remember from post 02 that the 32 threads of a warp have consecutive `threadIdx.x` values and run each load together. In this kernel, the 32 threads of a warp have 32 different rows and the same column. So in each step of the loop:

* The 32 reads of `A[row * n + k]` are `n` floats apart. They hit 32 different 32-byte sectors of memory.
* The 32 reads of `B[k * n + col]` are all the same address. That is one sector, broadcast to everyone.
* At the end, the 32 writes to `C` are again `n` floats apart.

Result: **662 GFLOP/s, 1.2% of cuBLAS**.

---

## 4. Step 2: Coalesced Access

{{< code file="cuda/post18/main.cu" region="coalesced" link="false" >}}

We only swap `x` and `y`. Now the threads of a warp share one row and have 32 consecutive columns:

* All 32 reads of `A` are the same address: one sector, broadcast.
* The 32 reads of `B` are 32 consecutive floats: 128 bytes, or 4 sectors.
* The writes to `C` are 128 consecutive bytes.

The compiled machine code (SASS) of steps 1 and 2 is identical: 2 global loads per fused multiply-add (FFMA). Only the addresses changed, and that alone makes the kernel **8.2x faster: 5,425 GFLOP/s**. This is post 07's naive kernel, and the lesson of post 05 in its purest form.

---

## 5. Step 3: Shared-Memory Tiles

{{< code file="cuda/post18/main.cu" region="smem" link="false" >}}

This is post 07's tiled kernel. A block of 32 × 32 threads loads a 32 × 32 tile of `A` and of `B` into shared memory, waits at `__syncthreads`, and then every thread reads from shared memory for 32 steps.

Global loads drop from 2 per FFMA to 2 per 32 FFMAs. Yet the kernel is only 18% faster: **6,423 GFLOP/s**. Post 07 explained why. The 4090's L1 and L2 caches were already catching most of step 2's repeated reads, and the new bottleneck is shared memory itself. Counting the SASS instructions of one tile (with `cuobjdump -sass`, below), each thread does 32 FFMAs and 40 shared-memory loads. That is 1.25 shared loads per FFMA. Shared memory is fast, but not that fast.

From here on, the goal of every step is the same: **do more math per load**.

---

## 6. Step 4: 1D Register Tiling

{{< code file="cuda/post18/main.cu" region="tile1d" link="false" >}}

Each thread now computes 8 results (`TM = 8`) in one column, instead of one. The block computes a 64 × 64 tile of `C` with 512 threads, and walks along `k` in thin slices of `BK = 8`.

The key is the inner loop. Each thread loads one value of `B` from shared memory, keeps it in a register, and uses it for 8 multiply-adds, one for each of its rows. The 8 accumulators `acc[TM]` also live in registers.

A few notes on the code:

* `A`, `B`, and `C` are moved to this block's tile at the start. This keeps the index math inside the loop short.
* 512 threads load a 64 × 8 tile of `A` and an 8 × 64 tile of `B`: exactly one float each.
* The compiler noticed that `As[trow + i][0..7]` is 8 consecutive floats and loads them with 128-bit instructions on its own.

In SASS, one tile is now 64 FFMAs and 24 shared loads: **0.375 shared loads per FFMA**, down from 1.25. The kernel jumps **3.1x, to 20,075 GFLOP/s**.

---

## 7. Step 5: 2D Register Tiling

If one dimension helps, two dimensions help more. Each thread now computes an 8 × 8 block of `C` (`TM = TN = 8`). The block tile grows to 128 × 128 with 256 threads.

{{< code file="cuda/post18/main.cu" region="tile2d" link="false" >}}

For every `k`, a thread loads 8 values of `A` into `a[]` and 8 values of `B` into `b[]`, then does their *outer product*: 8 × 8 = 64 multiply-adds from just 16 loaded values. In SASS, that is 64 FFMAs for 10 shared loads (8 scalar loads for `a[]`, and 2 128-bit loads the compiler made for `b[]`): **0.156 shared loads per FFMA**.

The cost is registers. A thread now holds 64 accumulators, and `ptxas` reports 122 registers per thread. As we saw in post 08, the SM has 65,536 registers, so it fits 2 blocks of 256 threads (16 warps) instead of 32 warps before. Fewer warps, but each one does far more useful work. Result: **29,975 GFLOP/s, 54% of cuBLAS**.

---

## 8. Step 6: 128-bit Loads

{{< code file="cuda/post18/main.cu" region="vec" link="false" >}}

Two changes, both about loading wider pieces at once:

1. **Global memory.** Each thread loads one `float4` (16 bytes) of `A` and one of `B` per tile, instead of 4 separate floats. That is 4x fewer load instructions.
2. **Shared memory.** In step 5, `a[]` needed 8 scalar loads, because the 8 values `As[trow + i][k]` are in a *column* of the tile, `BK` floats apart. So we store `A` **transposed** in shared memory: `As[k][m]`. Now the 8 values a thread needs for one `k` are next to each other, and two 128-bit loads fetch them.

The transpose happens for free while storing: the `float4` loaded from global memory is split into its four components and written to four rows of `As`. `B` needs no transpose, so it is copied as one `float4`.

The `reinterpret_cast<float4*>` lines look scary, but they only tell the compiler "these 4 floats are 16-byte aligned, load them in one instruction". That is true here, because `n`, `BK`, `trow`, and `tcol` are all multiples of 4.

In SASS, a tile is now 512 FFMAs and 32 shared loads, all 128-bit: **0.0625 shared loads per FFMA**. The kernel reaches **42,080 GFLOP/s, 76% of cuBLAS**.

---

## 9. Step 7: Double Buffering with `cp.async`

Look at the main loop of step 6 again. Every tile goes: load from global memory, `__syncthreads`, compute, `__syncthreads`. While the loads are in flight, the math units of this block have nothing to do. The fix is *double buffering*: keep two copies of each tile in shared memory, and load tile `t + 1` into one while computing tile `t` from the other.

Since the Ampere generation (compute capability 8.0), the GPU can copy from global memory straight into shared memory, without passing through registers, and without the thread waiting for it. In PTX this instruction is `cp.async`. CUDA C++ exposes it through three functions in `<cuda_pipeline.h>`:

| Function | What it does |
| --- | --- |
| `__pipeline_memcpy_async(dst, src, 16)` | Start copying 4, 8, or 16 bytes from global to shared memory. |
| `__pipeline_commit()` | Close the current group of copies. |
| `__pipeline_wait_prior(k)` | Wait until at most `k` of the latest groups are still in flight. |

It is the same idea as streams in post 12, but inside one kernel: start the work, keep going, and wait only when you need the result.

Here is how one tile is loaded:

{{< code file="cuda/post18/main.cu" region="load-tile" link="false" >}}

`cp.async` copies 16 bytes exactly as they are. It cannot split a `float4` into four rows like step 6 did, so `A` is back in its normal layout, `As[m][k]`. We also grew `BK` from 8 to 16, so each pair of `__syncthreads` covers twice as much math.

And here is the pipeline:

{{< code file="cuda/post18/main.cu" region="pipeline" link="false" >}}

Before the loop, we start loading tile 0 into buffer 0. In each iteration, we first start loading the *next* tile into the other buffer, and then wait with `__pipeline_wait_prior(1)`: "wait until at most one group, the one we just started, is still running". This means the current tile has arrived. After `__syncthreads`, every thread can see the whole tile. The second `__syncthreads` at the end makes sure nobody is still reading a buffer when the next iteration starts refilling it.

The result: **42,547 GFLOP/s**. About the same as step 6. Over several runs, steps 6 and 7 were always within a few percent of each other, in both directions.

Why didn't it help? The answer is in post 08: occupancy. `ptxas` reports 145 registers per thread for this kernel, against 123 for step 6. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` tells us what that costs:

| Step | Registers | Blocks per SM | Warps per SM |
| --- | --- | --- | --- |
| 6 float4 | 123 | 2 | 16 |
| 7 cp.async | 145 | 1 | 8 |

Registers are handed out in chunks, so 145 per thread × 256 threads no longer leaves room for a second block. And the second block was already doing our double buffering for us: while one block waited at `__syncthreads` for its loads, the other block computed. We added overlap inside the block but lost the overlap between blocks.

`cp.async` is still the right tool. It needs no registers to stage the data, and it makes deeper pipelines possible. cuBLAS itself, on this GPU, runs a CUTLASS kernel named `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`, which uses 4 stages. (You can see that name with `nsys profile --stats=true ./sgemm 8`.) But on its own, it only pays off if the rest of the kernel lets it. The next step does that.

---

## 10. Step 8: Warp Tiling

So far, we have assigned outputs to threads without thinking about warps: thread `tid` gets the 8 × 8 block at row `tid / 16 * 8`, column `tid % 16 * 8`. Now let's look at what one warp reads from shared memory in step 7.

Recall from post 06 that shared memory has 32 banks, each 4 bytes wide, and that two threads reading *different* addresses in the *same* bank must wait for each other.

* **Reads of `B`**: 16 threads read `float4`s at columns 0, 8, 16, ..., 120. They are 32 bytes apart, so half of every 128-byte row of banks is wasted. The warp needs twice as many shared-memory passes (*wavefronts*) as the data it actually gets.
* **Reads of `A`**: the warp reads two rows that are 8 rows apart. With `BK = 16`, that is 8 × 16 × 4 = 512 bytes apart, a multiple of 128 bytes. So both rows sit in the same banks: a 2-way bank conflict.

Warp tiling fixes this by deciding where each *warp* works first, and then where each thread works inside its warp:

{{< code file="cuda/post18/main.cu" region="warp-map" link="false" >}}

The 8 warps of a block are arranged as 4 rows × 2 columns, each owning a 32 × 64 piece of the 128 × 128 tile. Inside a warp, the 32 lanes form a 4 × 8 grid. A thread still computes 64 results, but spread out:

* 8 rows: `row0`, `row0 + 4`, ..., `row0 + 28`.
* 8 columns: `col0..col0+3` and `col0+32..col0+35`.

The inner loop is the same outer product as before, with the new indices:

{{< code file="cuda/post18/main.cu" region="warp-inner" link="false" >}}

Now look at one warp again:

* **Reads of `B`**: 8 lanes read 8 *consecutive* `float4`s: 128 bytes, all 32 banks, no waste. The other lanes read the same addresses and get them by broadcast.
* **Reads of `A`**: the warp reads 4 *consecutive* rows. We also pad each row of `As` from 16 to 20 floats (`BK + PAD`). The 4 rows then start at banks 0, 20, 8 (40 − 32), and 28 (60 − 32): four different groups of 4 banks, no conflict. The padding keeps each row 16-byte aligned, which `cp.async` needs. This is the `+1` padding trick from post 06, adapted to 16-byte loads.

As a bonus, the new layout also brought the kernel down to exactly 128 registers, so 2 blocks fit on an SM again.

Result: **43,623 GFLOP/s in the full run, 78.8% of cuBLAS**. Run on its own with `./sgemm 8`, it landed between 80% and 83% of cuBLAS in every one of my runs. When I ran `./sgemm 6`, `./sgemm 7`, and `./sgemm 8` in turn for five rounds, step 8 beat steps 6 and 7 every round. The margin is small, though: in one of my full runs, step 8 came out slightly behind step 6.

I also tried a few things that did *not* help on the 4090: 3 and 4 pipeline stages with `BK = 8`, a larger 256 × 128 block tile, and a different order of blocks for better L2 reuse. Each one was within noise of step 8 or slower. Getting the last 20% is where cuBLAS earns its keep. It uses bigger tiles with 128 results per thread, a 4-stage pipeline, and many small instruction-level tricks, and it picks among many kernels depending on the matrix shape.

---

## 11. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o sgemm -lcublas
./sgemm
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post18/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/sgemm
```
{{< /tab >}}
{{< /tabs >}}

As in post 07, we link cuBLAS with `-lcublas` or `CUDA::cublas`. `-arch=native` matters more than usual here: `cp.async` only exists on compute capability 8.0 and newer. On a Turing card (7.5), step 7 and step 8 still compile and run correctly, but the copies become ordinary synchronous loads.

Output: (RTX 4090)

```text
SGEMM 4096 x 4096 x 4096 (FP32)
kernel              ms   GFLOP/s    cuBLAS  max error
cuBLAS           2.482     55378    100.0%          -
1 naive        207.457       662      1.2%   4.12e-04
2 coalesced     25.336      5425      9.8%   4.12e-04
3 smem tile     21.398      6423     11.6%   4.12e-04
4 1D tile        6.846     20075     36.3%   4.12e-04
5 2D tile        4.585     29975     54.1%   4.12e-04
6 float4         3.266     42080     76.0%   4.12e-04
7 cp.async       3.230     42547     76.8%   4.12e-04
8 warp tile      3.151     43623     78.8%   4.12e-04
```

And step 8 on its own:

```bash
./sgemm 8
```

```text
SGEMM 4096 x 4096 x 4096 (FP32)
kernel              ms   GFLOP/s    cuBLAS  max error
cuBLAS           2.592     53029    100.0%          -
8 warp tile      3.171     43346     81.7%   4.12e-04
```

The numbers move by a few percent from run to run, for cuBLAS too. Do not read too much into the last digit.

All eight kernels have the same maximum error, `4.12e-04`. That is not a coincidence: every one of them adds the 4096 products in the same order, `k = 0` to `k = 4095`, so they produce bit-identical results. The difference is between that order and the order cuBLAS uses.

Since steps 7 and 8 juggle two shared-memory buffers, a missing `__syncthreads` would be an easy bug to make. I checked all the shared-memory kernels with `compute-sanitizer --tool racecheck` (with `n` temporarily set to 512 to keep it quick), and it reported 0 hazards.

---

## 12. The Scoreboard

Here is the whole climb in one table. The last columns hold the one number that best explains each jump. Shared loads per FFMA come from counting instructions in the compiled kernel; registers from `nvcc -Xptxas -v`; blocks per SM from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

| Step | GFLOP/s | % of cuBLAS | Speedup vs previous | What explains it |
| --- | ---: | ---: | ---: | --- |
| 1 naive | 662 | 1.2% | | Global loads: a warp touches ~16.5 sectors per request (expected) |
| 2 coalesced | 5,425 | 9.8% | 8.2x | ~2.5 sectors per request (expected); same SASS as step 1 |
| 3 smem tile | 6,423 | 11.6% | 1.18x | Global loads per FFMA: 2 → 0.0625; but 1.25 shared loads per FFMA |
| 4 1D tile | 20,075 | 36.3% | 3.1x | Shared loads per FFMA: 1.25 → 0.375 |
| 5 2D tile | 29,975 | 54.1% | 1.49x | Shared loads per FFMA: 0.375 → 0.156 |
| 6 float4 | 42,080 | 76.0% | 1.40x | Shared loads per FFMA: 0.156 → 0.0625, all 128-bit |
| 7 cp.async | 42,547 | 76.8% | 1.01x | 145 registers: blocks per SM 2 → 1 |
| 8 warp tile | 43,623 | 78.8% | 1.03x | Conflict-free shared loads (expected); 128 registers, 2 blocks per SM |
| cuBLAS | 55,378 | 100% | | |

From step 1 to step 8, the kernel got **66x faster**. From post 07's tiled kernel (step 3), it got **6.8x faster**.

The instruction counts are easy to get yourself. `cuobjdump -sass` prints the machine code of every kernel. Since most of our inner loops are fully unrolled, one tile's worth of work shows up as a straight line of instructions. `-fun` picks one kernel, but it wants the C++ *mangled* name, which the listing shows:

```bash
cuobjdump -sass sgemm | grep "Function :"
```

```text
		Function : _Z10sgemm_warpPKfS0_Pfi
		Function : _Z11sgemm_asyncPKfS0_Pfi
		Function : _Z9sgemm_vecPKfS0_Pfi
		Function : _Z8sgemm_2dPKfS0_Pfi
		Function : _Z8sgemm_1dPKfS0_Pfi
		Function : _Z10sgemm_smemPKfS0_Pfi
		Function : _Z15sgemm_coalescedPKfS0_Pfi
		Function : _Z11sgemm_naivePKfS0_Pfi
```

Then we count the FFMAs and the shared loads (`LDS`) of step 6:

```bash
cuobjdump -sass -fun _Z9sgemm_vecPKfS0_Pfi sgemm 2>/dev/null | grep -c FFMA
cuobjdump -sass -fun _Z9sgemm_vecPKfS0_Pfi sgemm 2>/dev/null | grep -c LDS
```

```text
512
32
```

---

## 13. Profiling the Steps with Nsight Compute

The numbers marked "expected" in the table come from counting addresses by hand, as we did in sections 3, 4, and 10. `ncu` can measure them directly. Our program takes a kernel number, so each profile is one command. `-k regex:sgemm_` skips the cuBLAS kernel, and `-c 1` profiles only the first launch:

```bash
# Steps 1 and 2: sectors per global load request
ncu -k regex:sgemm_ -c 1 \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./sgemm 1

# Steps 3 to 6: shared loads and FFMAs
ncu -k regex:sgemm_ -c 1 \
    --metrics smsp__inst_executed_op_shared_ld.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    ./sgemm 4

# Steps 7 and 8: bank conflicts and occupancy
ncu -k regex:sgemm_ -c 1 \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
    --section Occupancy \
    ./sgemm 7
```

Divide sectors by requests for steps 1 and 2. You should see about 16.5 and 2.5. For steps 7 and 8, the bank conflict count should drop to almost zero for step 8, with fewer wavefronts for the same work.

On the machine I used for this series, `ncu` stops with `ERR_NVGPUCTRPERM`, because reading the GPU's performance counters needs admin rights by default. So I could not capture these numbers here, and the table above uses only numbers I could measure without counters. Post 04 explains the fix: run `ncu` with `sudo`, or allow all users with the `NVreg_RestrictProfilingToAdminUsers=0` driver option and a reboot.

---

## 14. Exercise

Tune the slice depth `BK` of step 8. It lives in the `twp` namespace:

```cpp
constexpr int BM = 128, BN = 128, BK = 16, PAD = 4;
```

1. Set `BK = 8`. Build and run `./sgemm 8`. On my 4090, it dropped to about 37,400 GFLOP/s (68% of cuBLAS). Why does a thinner slice hurt? Count how many `__syncthreads` each thread executes for the whole multiply.
2. Set `BK = 32`. This time it does not even compile:

   ```text
   ptxas error   : Entry function '_Z10sgemm_warpPKfS0_Pfi' uses too much shared data (0x11000 bytes, 0xc000 max)
   ```

   Two buffers of 128 × 36 plus 32 × 128 floats is 69,632 bytes, and a kernel can only declare 48 KB (`0xc000`) of static shared memory. The 4090 has up to 99 KB per block, but above 48 KB you must use *dynamic* shared memory: declare `extern __shared__ float smem[];`, pass the size as the third launch parameter `<<<grid, block, bytes>>>`, and allow it once with `cudaFuncSetAttribute(sgemm_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. Make it work and measure. Is it faster? How many blocks per SM fit now?

In summary, we took a matrix multiply from 662 to about 44,000 GFLOP/s, roughly 80% of cuBLAS, one idea at a time. Coalescing made each warp's loads land in few sectors. Shared memory and then register tiling cut the loads needed per multiply-add from 2 down to about 0.06, which moved the bottleneck from memory to math. Wider `float4` loads cut instructions further. `cp.async` double buffering did not help by itself, because it cost a second resident block, a good reminder that every optimization must be measured. Warp tiling made shared-memory reads conflict-free and gave us the final few percent. The remaining gap to cuBLAS is real engineering, bigger tiles, deeper pipelines, and instruction scheduling, and it is exactly what libraries like cuBLAS and CUTLASS exist to do for us.

[1]: https://siboehm.com/articles/22/CUDA-MMM "Simon Boehm: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies "CUDA C++ Programming Guide: Asynchronous Data Copies"
[3]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "Nsight Compute Profiling Guide"
[4]: https://github.com/NVIDIA/cutlass "CUTLASS"
