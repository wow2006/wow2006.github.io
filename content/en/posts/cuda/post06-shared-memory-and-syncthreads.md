---
title: "Post 06: Shared memory and __syncthreads"
date: 2026-09-29T19:06:00+03:00
weight: 7
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 05, we transposed a matrix and hit a wall. A warp can read a row of the matrix in one go, because 32 neighbouring threads touch 32 neighbouring floats. But a transpose turns rows into columns, so whichever side we make coalesced, the other side jumps by a whole row between threads. Our naive transpose reached less than a third of the copy speed.

The fix is to take a detour through a small, fast memory that lives inside each SM: *shared memory*. A block reads a tile of the matrix row by row, keeps it in shared memory, and then writes it out row by row in the transposed position. Both global accesses become coalesced, and the "turning" happens where jumping around is cheap. In this post, we will build that kernel, learn why it needs `__syncthreads`, and fix a quiet slowdown called a *bank conflict* with a single extra column.

## What We Will Learn

In this article, we will cover:

* What shared memory is, and how to declare it with `__shared__`.
* How to transpose a matrix through a 32 × 32 tile in shared memory.
* Why the threads of a block must wait for each other with `__syncthreads`.
* How to catch a missing `__syncthreads` with `compute-sanitizer --tool racecheck`.
* What shared memory banks are, and how a `[32][33]` array avoids bank conflicts.

## Table of Contents

1. The Full Program.
2. Shared Memory.
3. The Tiled Transpose.
4. Why We Need `__syncthreads`.
5. Build and Run.
6. Catching the Race with `racecheck`.
7. Bank Conflicts and the `+1` Trick.
8. Seeing Bank Conflicts with `ncu`.
9. Exercise.

---

## 1. The Full Program

```bash
mkdir transpose
cd transpose
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post06/main.cu" >}}

We keep `CUDA_CHECK`, `make_device_buffer`, and the `time_ms` helper from post 03. The program runs four kernels on an 8192 × 8192 matrix of floats and prints the score of each one:

{{< code file="cuda/post06/main.cu" region="launch" link="false" >}}

Each kernel reads the matrix once and writes it once, so the useful traffic is `2 * bytes`. After timing, we copy the result back and check every element against the CPU.

The matrix is 256 MB, and we need two of them. That is much bigger than the 72 MB L2 cache of the RTX 4090, so, as we saw in post 03, the numbers measure the real memory and not the cache. We will break this rule on purpose later, and you will see why.

---

## 2. Shared Memory

So far, every kernel read and wrote *global memory*, the large VRAM we allocate with `cudaMalloc`. It is big, but it is far away: a load takes hundreds of clock cycles.

Each SM also has a small memory on the chip itself. A kernel can use part of it as *shared memory*. It has three important properties:

* It is fast: roughly as fast as the L1 cache, because it is the same hardware.
* It is small: tens of kilobytes per block, not gigabytes.
* It belongs to a block. All threads of a block see the same shared memory, and threads of other blocks cannot see it. It is gone when the block finishes.

We declare it inside a kernel with the `__shared__` keyword:

```cpp
__shared__ float tile[32][32];
```

This looks like a local array, but it is not one per thread. There is exactly one `tile` per block, and all 256 threads of the block read and write the same array. The size must be known at compile time. Here it is 32 × 32 × 4 bytes = 4 KB.

Because the threads of a block share it, shared memory is how threads cooperate. One thread can load a value from global memory, and another thread can use it.

---

## 3. The Tiled Transpose

First, here are the two kernels we compare against. `copy_2d` moves the matrix without transposing it. It has the same shape as a transpose, so it is the ceiling: a transpose cannot be faster than a copy. `transpose_naive` is `transpose_read_coalesced` from post 05 under a shorter name, with coalesced reads and strided writes:

{{< code file="cuda/post06/main.cu" region="naive" link="false" >}}

All kernels in this post use the same layout. Each block covers a 32 × 32 tile of the matrix with 32 × 8 = 256 threads, so each thread handles 4 rows of the tile, 8 rows apart. With 32 threads along `x`, one warp covers exactly one row of 32 floats.

Now the shared-memory version:

{{< code file="cuda/post06/main.cu" region="shared" link="false" >}}

Ignore the `Pad` template parameter for now; it is `0` until section 7. The kernel works in two phases.

**Load.** The block reads its tile from the input, row by row. Thread `(tx, ty)` stores the value from row `y + j` and column `x` into `tile[ty + j][tx]`. Neighbouring threads read neighbouring addresses, so every read is coalesced.

**Store.** The block writes the tile to the mirrored position in the output. The block at tile `(bx, by)` of the input writes to tile `(by, bx)` of the output, which is why the two `blockIdx` are swapped. Again, neighbouring threads write neighbouring addresses, so every write is coalesced too. The transpose itself happens inside the tile: thread `(tx, ty)` reads `tile[tx][ty + j]`, a *column* of the tile.

So the strided access did not disappear. We moved it from global memory, where it costs whole memory transactions, to shared memory, where it is cheap. Well, almost cheap; section 7 is about that.

---

## 4. Why We Need `__syncthreads`

Look at which thread writes the value that thread `(tx, ty)` reads. In the store phase, it reads `tile[tx][ty + j]`. That element was written in the load phase by the thread whose `threadIdx.x` is `ty + j`, and that is almost never the same thread. Usually it is not even in the same warp.

The GPU does not run the 8 warps of a block in lock step. One warp may finish its loads and move on to the store phase while another warp is still waiting for its data to arrive from global memory. If the first warp reads the tile at that moment, it gets whatever was there before: old values from a previous block, or garbage.

`__syncthreads()` is a barrier for the block. No thread passes it until every thread of the block has reached it. After the barrier, all writes to shared memory made before it are visible to all threads of the block. So the kernel is split cleanly: first everyone loads, then everyone stores.

There is one rule to keep in mind. Every thread of the block must reach the same `__syncthreads()`. If you put it inside an `if` that only some threads enter, the block can hang or behave in undefined ways. Here it is at the top level of the kernel, so all threads reach it.

---

## 5. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o transpose
./transpose
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post06/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/transpose
```
{{< /tab >}}
{{< /tabs >}}

Output: (RTX 4090)

```text
Matrix 8192 x 8192 (256 MB), peak 1008 GB/s
kernel                time (ms)      GB/s   of peak
copy_2d (ceiling)        0.5827     921.4     91.4%   PASSED
transpose_naive          1.8677     287.4     28.5%   PASSED
transpose_shared<0>      0.6508     824.9     81.8%   PASSED
transpose_shared<1>      0.6502     825.7     81.9%   PASSED
```

The shared-memory transpose is 2.9 times faster than the naive one, and it reaches 90% of the copy. That is the whole point of this post in one line: when one side of the access pattern is strided, stage the data in shared memory and make both sides coalesced.

The remaining 10% to the copy is most likely the barrier. While a block waits at `__syncthreads`, its warps issue no memory requests. Other blocks on the same SM fill some of that gap, but not all of it.

You may also notice that `transpose_shared<1>`, the padded version, is no faster here. Keep that in mind; we will come back to it.

---

## 6. Catching the Race with `racecheck`

Let's see what happens without the barrier. Comment out the line in `transpose_shared`:

```cpp
    // __syncthreads();
```

Build and run again:

```text
Matrix 8192 x 8192 (256 MB), peak 1008 GB/s
kernel                time (ms)      GB/s   of peak
copy_2d (ceiling)        0.5826     921.5     91.4%   PASSED
transpose_naive          1.8812     285.4     28.3%   PASSED
transpose_shared<0>      0.6515     824.0     81.7%   FAILED (19104275 wrong elements)
transpose_shared<1>      0.6508     824.5     81.8%   FAILED (19331785 wrong elements)
```

About 19 million of the 67 million elements are wrong, and the count changes from run to run. This is a *race condition*: the result depends on which warp happens to run first. Here we were lucky that our check caught it. With a different block shape, a smaller matrix, or another GPU, the race might fail only once in a thousand runs.

In post 01, we used `compute-sanitizer` to find out-of-bounds accesses. It has a separate tool for races in shared memory, `racecheck`. It is slow, so we give it a small 64 × 64 matrix. We also build with `-lineinfo`, so that it can print source lines:

```bash
nvcc -U_GNU_SOURCE -arch=native -O3 -lineinfo main.cu -o transpose
compute-sanitizer --tool racecheck ./transpose 64
```

Output: (shortened)

```text
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at void transpose_shared<(int)0>(const float *, float *, int)+0x190 in main.cu:86
=========     and Read access at void transpose_shared<(int)0>(const float *, float *, int)+0x1d0 in main.cu:96 [896 hazards]
=========     and Read access at void transpose_shared<(int)0>(const float *, float *, int)+0x1e0 in main.cu:96 [1024 hazards]
=========     and Read access at void transpose_shared<(int)0>(const float *, float *, int)+0x1f0 in main.cu:96 [1024 hazards]
=========     and Read access at void transpose_shared<(int)0>(const float *, float *, int)+0x200 in main.cu:96 [1024 hazards]
=========
...
Matrix 64 x 64 (0 MB), peak 1008 GB/s
kernel                time (ms)      GB/s   of peak
copy_2d (ceiling)        5.1344       0.0      0.0%   PASSED
transpose_naive          5.0957       0.0      0.0%   PASSED
transpose_shared<0>     13.9563       0.0      0.0%   PASSED
transpose_shared<1>      7.0441       0.0      0.0%   PASSED
========= RACECHECK SUMMARY: 100 hazards displayed (184 errors, 0 warnings)
```

Line 86 is the write into the tile, `tile[threadIdx.y + j][threadIdx.x] = ...`, and line 96 is the read, `... = tile[threadIdx.x][threadIdx.y + j]`. The four read addresses are the four iterations of the unrolled `j` loop. That is exactly the pair of accesses the barrier is supposed to separate.

Look at the program output under the tool: everything `PASSED`. The sanitizer changes the timing so much that the race did not produce a wrong value this time. It reported the hazard anyway, because `racecheck` does not wait for a wrong result; it watches every shared-memory access and flags a read and a write to the same address from different threads with no barrier between them. That is why it is worth running even when your tests pass.

Put `__syncthreads()` back, rebuild, and run it again:

```text
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

(The timings in these runs are meaningless; the tool slows every kernel down by orders of magnitude.)

---

## 7. Bank Conflicts and the `+1` Trick

Shared memory is split into 32 *banks*. Each bank is 4 bytes wide, and consecutive 4-byte words go to consecutive banks:

```text
word address:  0   1   2  ...  31  32  33  ...  63  64 ...
bank:          0   1   2  ...  31   0   1  ...  31   0 ...
```

In other words, `bank = word_index % 32`. Each bank can serve one word per clock. When the 32 threads of a warp access 32 different banks, the access takes one step. When several threads access *different* words in the *same* bank, the hardware splits the access into several steps, one per word. That is a *bank conflict*. If all 32 threads hit the same bank, it is a 32-way conflict, and the access is 32 times slower. (Threads reading the *same* word is fine; the value is broadcast.)

Now apply this to our tile. With `float tile[32][32]`, element `tile[r][c]` is word `r * 32 + c`, so it lives in bank `(r * 32 + c) % 32 = c`. The bank depends only on the column.

* **Load phase:** a warp writes `tile[ty + j][tx]` for `tx = 0..31`. Same row, 32 different columns, so 32 different banks. No conflict.
* **Store phase:** a warp reads `tile[tx][ty + j]` for `tx = 0..31`. 32 different rows, same column, so all 32 threads hit the same bank. A 32-way conflict, on every read.

The fix is one extra column that we never use:

```cpp
__shared__ float tile[32][33];
```

Now `tile[r][c]` is word `r * 33 + c`, which lives in bank `(r * 33 + c) % 32 = (r + c) % 32`. Walking down a column, each row shifts the bank by one, so the 32 reads of the store phase land in 32 different banks. The load phase is still fine, too. We pay 32 × 4 = 128 bytes of shared memory per block for it.

In our program, that is the `Pad` template parameter: `transpose_shared<0>` is the `[32][32]` version and `transpose_shared<1>` is the `[32][33]` version.

But our measurement said the padding made no difference. Why?

Because the kernel is limited by global memory, not by shared memory. The whole transpose takes 0.65 ms, and almost all of it is spent waiting for DRAM. The extra steps for the bank conflicts happen while the warps would be waiting anyway, so they are hidden.

To see the conflicts, we have to take DRAM out of the picture. The program takes the matrix size as an argument. A 2048 × 2048 matrix is 16 MB, so both matrices fit in the 72 MB L2 cache. As we learned in post 03, this no longer measures memory bandwidth; the "percent of peak" is meaningless here. But it makes the memory much faster, and whatever is slow inside the kernel shows up:

```bash
./transpose 2048
```

```text
Matrix 2048 x 2048 (16 MB), peak 1008 GB/s
kernel                time (ms)      GB/s   of peak
copy_2d (ceiling)        0.0095    3523.4    349.5%   PASSED
transpose_naive          0.1099     305.4     30.3%   PASSED
transpose_shared<0>      0.0166    2027.0    201.1%   PASSED
transpose_shared<1>      0.0093    3603.4    357.4%   PASSED
```

Now the difference is clear. The `[32][32]` version takes 1.8 times longer than the padded one, and the padded one runs as fast as the copy. Also note that the naive transpose stays at about 300 GB/s even from the L2 cache: its strided writes waste most of every memory transaction, no matter where the data lives.

So is the padding worth it? Yes. It costs nothing, and in real kernels, shared memory is rarely the only thing a kernel does. Once the global memory part gets faster (or the data is already in the cache), conflicts become the bottleneck. It is a good habit to check the access pattern of every shared-memory array for conflicts.

---

## 8. Seeing Bank Conflicts with `ncu`

Timing tells us something is slow. Nsight Compute can tell us it is bank conflicts, by counting them. These are the metrics:

| Metric | Meaning |
| ------ | ------- |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Extra steps caused by conflicts on shared-memory loads |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | The same, for stores |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | Total steps (*wavefronts*) for shared-memory loads |

Our program launches each kernel 23 times (3 warm-up + 20 timed). With `-k` we select the two `transpose_shared` kernels, skip the first 22 matches, and profile two: the last launch of `transpose_shared<0>` and the first of `transpose_shared<1>`:

```bash
ncu -k regex:transpose_shared --launch-skip 22 --launch-count 2 \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
    ./transpose 2048
```

On the machine I used for this series, this fails with `ERR_NVGPUCTRPERM`: normal users are not allowed to read the GPU performance counters. We covered the fix in post 04: either run `ncu` as root with its full path (`sudo /usr/local/cuda/bin/ncu ...`), or set `NVreg_RestrictProfilingToAdminUsers=0` for the driver and reboot. I did not have root access for this run, so I cannot show you the real numbers. Instead, here is what the arithmetic predicts, so you can check it on your machine.

For a 2048 × 2048 matrix, there are 64 × 64 = 4096 blocks, each with 8 warps, and each warp does 4 shared-memory loads. That is 131,072 warp-wide loads per launch.

* `transpose_shared<0>`: each load is a 32-way conflict, so it takes 32 wavefronts instead of 1. Expect about 131,072 × 32 = 4.2 million load wavefronts and about 131,072 × 31 = 4.1 million bank conflicts.
* `transpose_shared<1>`: each load takes one wavefront. Expect about 131,072 load wavefronts and conflicts close to zero.
* Stores: close to zero conflicts in both kernels, because the load phase writes rows.

If you open the full report instead (`ncu --set full`), the *Memory Workload Analysis* section shows the same information as a table of shared-memory wavefronts and bank conflicts, and the *Source* page can point at the exact line.

---

## 9. Exercise

The `+1` works because 33 and 32 share no common factor, so walking down a column visits every bank. What about other paddings? Add these lines to the `kernels` table:

```cpp
        {"transpose_shared<2>", transpose_shared<2>, true},
        {"transpose_shared<16>", transpose_shared<16>, true},
        {"transpose_shared<32>", transpose_shared<32>, true},
```

Before you run it, work out the bank of `tile[r][c]` for each width (34, 48, and 64 words per row) as `r` goes from 0 to 31. How many different banks does a column touch, and how many threads land on each one? Then run `./transpose 2048` and compare with your prediction.

Here is what I got:

```text
Matrix 2048 x 2048 (16 MB), peak 1008 GB/s
kernel                time (ms)      GB/s   of peak
copy_2d (ceiling)        0.0102    3275.8    324.9%   PASSED
transpose_naive          0.1182     283.8     28.2%   PASSED
transpose_shared<0>      0.0179    1872.6    185.8%   PASSED
transpose_shared<1>      0.0100    3339.9    331.3%   PASSED
transpose_shared<2>      0.0100    3361.9    333.5%   PASSED
transpose_shared<16>     0.0115    2922.9    289.9%   PASSED
transpose_shared<32>     0.0180    1868.6    185.4%   PASSED
```

A width of 34 gives a 2-way conflict, 48 a 16-way conflict, and 64 a 32-way conflict, the same as no padding at all. Notice that the 2-way conflict does not show up in the time. The 16-way one does, and the 32-way one costs the most. If you have `ncu` access, count the conflicts with the command from section 8, but drop `--launch-skip` and `--launch-count`. `ncu` then profiles all 23 launches of each kernel, which takes a while, but every launch gives the same counts. See how they follow the arithmetic.

In summary, shared memory is a small, fast memory that the threads of a block share. We used it to stage a 32 × 32 tile so that both the reads and the writes of a transpose are coalesced, which made it 2.9 times faster than the naive version and brought it to 90% of a plain copy. Because threads read what other threads wrote, we need `__syncthreads()` between the two phases; without it, the result is silently wrong, and `compute-sanitizer --tool racecheck` finds the race even when the output looks right. Finally, shared memory is split into 32 banks, and a column of a `[32][32]` float array lives entirely in one of them. One extra column, `[32][33]`, spreads it across all 32 banks for free. In the next post, we will use the same tiles for the most important kernel of all: matrix multiplication.

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory "CUDA C++ Programming Guide: Shared Memory"
[2]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory "CUDA C++ Best Practices Guide: Shared Memory"
[3]: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ "An Efficient Matrix Transpose in CUDA C/C++"
[4]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool "Compute Sanitizer: Racecheck Tool"
[5]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "Nsight Compute Profiling Guide"
