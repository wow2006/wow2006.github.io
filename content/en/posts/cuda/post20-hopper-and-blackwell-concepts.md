---
title: "Post 20: Hopper and Blackwell Concepts"
date: 2026-09-29T19:20:00+03:00
weight: 21
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 18 we built an SGEMM step by step, and the last steps were all about feeding the Tensor Cores: copy a tile into shared memory with `cp.async`, wait for it, run `mma.sync`, repeat. Every thread in the block took part in the copies, and every warp kept its own accumulators in registers. On the RTX 4090 that is as far as the hardware goes.

Data-center GPUs moved on. Hopper (H100, 2022) and data-center Blackwell (B200, 2024) added hardware that does those copies for us, lets blocks read each other's shared memory, and runs matrix multiplies that are issued by a whole warpgroup, or even by a single thread. This is what cuBLAS, CUTLASS and FlashAttention 3/4 use on those GPUs today.

None of it runs on a 4090. So this post is different from the others: it is a map of the concepts, with short snippets that we compile for the new architectures but cannot run here. Every snippet is labelled with the architecture it needs.

## What We Will Learn

In this article, we will cover:

* Thread block clusters and distributed shared memory (DSMEM).
* TMA: the copy engine that moves whole tiles, and the `mbarrier` that tells us when they arrived.
* `wgmma`, Hopper's warpgroup matrix multiply.
* `tcgen05` and tensor memory (TMEM) on data-center Blackwell.
* Which of these the RTX 50 series (sm_120) has, and which it does not.
* How to reach all of this without writing PTX: CUTLASS/CuTe in C++ and the CuTe Python DSL.

## Table of Contents

1. The Feature Map.
2. Thread Block Clusters.
3. Distributed Shared Memory.
4. TMA: Tile Copies Done by Hardware.
5. wgmma on Hopper.
6. tcgen05 and Tensor Memory on Blackwell.
7. What About the RTX 50 Series?
8. Where to Reach Them: CUTLASS and the CuTe DSL.
9. Compile Check.
10. Exercise.

---

## 1. The Feature Map

Before the details, here is the whole post in one table. "First arch" is the compute capability where the feature appeared. The last column is the one that matters for us.

| Feature | First arch | What it replaces | On a 4090 (sm_89)? |
| ------- | ---------- | ---------------- | ------------------ |
| Thread block clusters (`__cluster_dims__`) | sm_90 (Hopper) | Blocks that can only talk through global memory | No |
| Distributed shared memory (DSMEM) | sm_90 | A round trip through global memory / L2 | No |
| TMA (`cp.async.bulk.tensor`) + transaction `mbarrier` | sm_90 | Every thread issuing its own loads or `cp.async` (post 18) | No |
| `wgmma` (warpgroup MMA) | sm_90a only | `mma.sync` per warp (post 17) | No |
| `tcgen05.mma` + tensor memory (TMEM) | sm_100a / sm_103a (data-center Blackwell) | `wgmma` with accumulators in registers | No |

Two things in that table need a word of explanation.

The `a` in `sm_90a` and `sm_100a` means *architecture-specific*. Normal code compiled for `sm_90` will also run on later GPUs. Code that uses `wgmma` or `tcgen05` is compiled for `sm_90a` or `sm_100a`, and it runs **only** on that exact family. That is why `wgmma` is Hopper-only: Blackwell does not have it, and CUDA 13.1's `ptxas` refuses it for `sm_100a`:

```text
error   : Instruction 'wgmma.fence' not supported on .target 'sm_100a'
```

The second point: "data-center Blackwell" is important. B200/GB200 are `sm_100`, B300/GB300 are `sm_103`. The RTX 50 series is also called Blackwell, but it is `sm_120`, and it is missing the most interesting row of the table. We come back to this in section 7.

---

## 2. Thread Block Clusters

> sm_90+, not run here.

Up to now, a block was an island. Blocks could not wait for each other, and the only way to share data was through global memory. The hardware groups SMs into *GPU processing clusters* (GPCs), but CUDA never let us use that.

Hopper adds a new level between the block and the grid: the *cluster*. A cluster is a group of blocks (up to 8 portably, 16 on H100 with an opt-in) that the hardware promises to run **at the same time**, on SMs in the same GPC. Because they are guaranteed to be running together, they can synchronize with each other and read each other's shared memory.

The simplest way to make a cluster is an attribute on the kernel:

{{< code file="cuda/post20/main.cu" region="cluster" link="false" >}}

`__cluster_dims__(4, 1, 1)` says: every 4 consecutive blocks along `x` form one cluster. The launch itself does not change; we still write `cluster_sum<<<64, 256>>>(...)`, and the grid size must be a multiple of the cluster size.

Inside the kernel, `cooperative_groups::this_cluster()` gives us a handle, just like `this_thread_block()` from post 09:

* `cluster.block_rank()` is our block's index inside the cluster (0 to 3 here).
* `cluster.num_blocks()` is the cluster size.
* `cluster.sync()` is a `__syncthreads()` for the whole cluster: every thread of every block in the cluster waits there.

If the cluster size is only known at run time, there is no attribute. We use `cudaLaunchKernelEx` with a launch attribute instead:

```cpp
cudaLaunchConfig_t config = {};
config.gridDim = blocks;
config.blockDim = threads;

cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim = {4, 1, 1};
config.attrs = &attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, my_kernel, args...);
```

---

## 3. Distributed Shared Memory

> sm_90+, not run here.

Look at the kernel above again. Each block sums its 256 elements into `partial`, a variable in its own shared memory. Then block 0 of the cluster does this:

```cpp
float* remote = cluster.map_shared_rank(&partial, r);
total += *remote;
```

`map_shared_rank` takes the address of a shared variable and a block rank, and returns a pointer to **the same variable in that other block's shared memory**, on a different SM. We can read, write and even use atomics on it. The shared memory of all blocks in the cluster together is called *distributed shared memory* (DSMEM). It travels over a dedicated SM-to-SM network, not through L2 or DRAM.

Without DSMEM, the same kernel would write four partial sums to global memory and launch a second kernel (or use a grid-wide sync) to add them. With it, the four partials never leave the SMs.

There are two `cluster.sync()` calls, and both are needed:

1. The first one makes sure every block has finished its `atomicAdd`s before block 0 reads their `partial`.
2. The second one is easy to forget. Shared memory belongs to a block, and it is freed when the block exits. If block 3 exits while block 0 is still reading its `partial`, block 0 reads memory that no longer exists. So every block waits at the end until block 0 is done.

The rule is: **a block must not exit while another block of the cluster may still access its shared memory.**

---

## 4. TMA: Tile Copies Done by Hardware

> sm_90+, not run here.

In post 18, loading a 32×32 tile into shared memory meant each of the 256 threads computed an address and issued its own `cp.async`. That costs registers and instructions in every thread, and all of it is address arithmetic, not math.

The *Tensor Memory Accelerator* (TMA) is a copy engine inside each SM. We describe the whole matrix to it once, and after that **one thread** can say "copy the tile at (x, y) into this shared memory buffer". The TMA computes the addresses, handles the edges of the matrix, and can even swizzle the layout on the way to avoid bank conflicts (post 06).

### The tensor map

The description is a 128-byte object called a `CUtensorMap`. We build it on the host with the driver API function `cuTensorMapEncodeTiled`:

{{< code file="cuda/post20/main.cu" region="tensor-map" link="false" >}}

It holds everything the TMA needs: the data type, the base address, the size of each dimension, the stride between rows in bytes, and the *box*, which is the tile size one copy moves. Dimensions are listed fastest-changing first, so for a row-major matrix that is `{cols, rows}`. Because it is a driver API function, we have to link `libcuda` (`-lcuda`).

### The kernel

{{< code file="cuda/post20/main.cu" region="tma-kernel" link="false" >}}

The tensor map is passed as a kernel parameter marked `const __grid_constant__`. That keeps it in the kernel's parameter space, where the TMA can read it; copying it into a local variable would break that.

The interesting part is how we know when the copy has finished. The TMA works asynchronously, so we need a barrier that understands *bytes*, not just threads. That is the `mbarrier`, a small object that lives in shared memory. Here we use it through `cuda::barrier` from libcu++ (post 11):

1. `init(&bar, blockDim.x)` says 128 threads must arrive.
2. Thread 0 starts the copy with `cp_async_bulk_tensor_2d_global_to_shared`, then calls `barrier_arrive_tx(bar, 1, sizeof(tile))`: "I arrived, and 4096 bytes are on their way".
3. The other threads just `arrive()`.
4. `bar.wait()` returns only when **both** all 128 threads have arrived **and** the TMA has delivered all 4096 bytes. The TMA decrements the byte count itself as data lands.

After that, the tile is in shared memory and every thread can use it. Writing it back is the same idea in reverse: one thread calls `cp_async_bulk_tensor_2d_shared_to_global`, then waits for the bulk group to finish reading.

The `fence_proxy_async_shared_cta()` calls are the one new rule. The TMA is a separate unit (an *async proxy*), and it does not automatically see what normal threads wrote to shared memory. A proxy fence makes our writes visible to it: once after initializing the barrier, and once before the TMA reads our modified tile.

In a real GEMM, this is how Hopper kernels are organised: a *producer* warp does nothing but issue TMA copies for the next tiles, and *consumer* warps do the math. This is called *warp specialization*, and it replaces the double-buffering from post 18.

---

## 5. wgmma on Hopper

> sm_90a only, not run here.

In post 17, `mma.sync` ran on one warp: 32 threads, a small tile (like 16×8×16), operands in registers, and the warp waits until it is done.

Hopper's `wgmma.mma_async` works on a *warpgroup*: 4 consecutive warps, 128 threads. It is bigger (M is always 64, N goes up to 256), it reads A and B **directly from shared memory** through 64-bit *matrix descriptors* instead of registers, and, as the name says, it is asynchronous. The accumulators are still in registers, spread across the 128 threads.

Here is the smallest one, `m64n8k16` with BF16 inputs and FP32 accumulators, in inline PTX. It assembles for `sm_90a` with CUDA 13.1:

```cpp
// sm_90a only. Not run here.
float d0 = 0, d1 = 0, d2 = 0, d3 = 0;   // 64x8 floats / 128 threads = 4 each
asm volatile("wgmma.fence.sync.aligned;");
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "l"(desc_a), "l"(desc_b));        // descriptors point into shared memory
asm volatile("wgmma.commit_group.sync.aligned;");
asm volatile("wgmma.wait_group.sync.aligned 0;");
```

The pattern is: fence, issue one or more `wgmma`s, commit them as a group, then later wait for the group. Between the commit and the wait, the warps are free to do other work, like issuing the TMA copy for the next tile. Building the descriptors (base address, leading and stride byte offsets, swizzle mode) is fiddly, and that is the main reason people use CUTLASS for this instead of raw PTX. A realistic kernel uses `m64n256k16`, which needs 128 accumulator registers per thread.

---

## 6. tcgen05 and Tensor Memory on Blackwell

> sm_100a / sm_103a only (data-center Blackwell), not run here.

On Hopper the accumulators still live in registers. A 64×256 FP32 tile per warpgroup is 128 registers per thread, which leaves little room for anything else and puts a ceiling on the tile size.

Data-center Blackwell moves the accumulators out of the register file into a new memory next to the Tensor Cores: **tensor memory** (TMEM). Each SM has 128 lanes × 512 columns of 32-bit cells, which is 256 KB. The instruction family that uses it is called `tcgen05` (5th-generation Tensor Core).

The main changes compared to `wgmma`:

* **TMEM must be allocated.** One warp calls `tcgen05.alloc` for a number of columns (a power of two, at least 32) and gets back a TMEM address. It must be freed with `tcgen05.dealloc` before the block exits.
* **One thread issues the MMA.** Not a warp, not a warpgroup: a single thread launches `tcgen05.mma`. A and B come from shared memory (descriptors, like `wgmma`), and D is accumulated in TMEM.
* **Completion goes to an mbarrier.** `tcgen05.commit` makes the mbarrier from section 4 fire when the MMA is done, so the same barrier machinery serves both TMA and MMA.
* **Results are read back explicitly.** When the tile is finished, warps copy it from TMEM into registers with `tcgen05.ld` for the epilogue. Each warp can reach only its own quarter of the 128 lanes.
* **Two SMs can work as one.** With `cta_group::2`, a pair of blocks in the same cluster (on two SMs) run one larger MMA together. This is where clusters from section 2 pay off.

Here is the skeleton, in inline PTX. It assembles for `sm_100a` with CUDA 13.1 (it only shows the shape; real code also needs the descriptors and the commit to an mbarrier):

```cpp
// sm_100a / sm_103a only. Not run here.
__shared__ uint32_t tmem_base;
if (threadIdx.x < 32) {  // one full warp allocates
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_base));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"(dst));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__syncthreads();
uint32_t d_tmem = tmem_base;       // TMEM address of our 128 x 32 accumulator
if (threadIdx.x == 0) {            // ONE thread issues the MMA
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"   // p = false: overwrite D
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
        :: "r"(d_tmem), "l"(desc_a), "l"(desc_b), "r"(idesc));
}
// ... tcgen05.commit -> mbarrier, wait, tcgen05.ld into registers ...
if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(d_tmem));
}
```

Put together, a Blackwell GEMM has three kinds of workers: one warp issues TMA loads, one thread issues `tcgen05.mma`, and a few warps move finished tiles out of TMEM. The math units are kept busy by very few threads, and the register file is free for the epilogue.

---

## 7. What About the RTX 50 Series?

This was an open question for this post: do consumer Blackwell GPUs (RTX 5090 and friends, `sm_120`) have `tcgen05` and TMEM? **No.** We can check it ourselves with the CUDA 13.1 compiler we already have, without owning the GPU. The skeleton above assembles for `sm_100a`, `sm_100f` and `sm_103a`, and fails for `sm_120a`, `sm_120f` and `sm_121a` (DGX Spark):

```text
error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
error   : Feature '.cta_group::1' not supported on .target 'sm_120a'
```

Community documentation says the same: sm_120 has no TMEM and keeps a `mma.sync`-style warp-level Tensor Core instruction set (extended with FP4/FP6/FP8 block-scaled formats), and a `tcgen05` kernel has to be rewritten for it, not recompiled [8][9]. It has no `wgmma` either, since that is Hopper-only.

What the RTX 50 series **does** have is the sm_90 baseline: our cluster and TMA program from sections 2–4 compiles for `sm_120` without errors. So on a consumer card the modern recipe is TMA + clusters + warp-level MMA, and on a data-center card it is TMA + clusters + `tcgen05`. (Checked 2026-09-29 with CUDA 13.1 `ptxas`.)

---

## 8. Where to Reach Them: CUTLASS and the CuTe DSL

Writing descriptors, swizzles and barrier phases by hand is how you learn, not how you ship. In practice there are two doors [5]:

**CUTLASS 3.x/4.x in C++.** CUTLASS is NVIDIA's open-source template library for GEMMs and convolutions, and CuTe is its layer for describing layouts and tiles. The TMA copies, `wgmma`, `tcgen05`, clusters and warp specialization are all wrapped there. Good starting points in the repository are `examples/48_hopper_warp_specialized_gemm` and `examples/70_blackwell_gemm`, and `examples/73_blackwell_gemm_preferred_cluster` for clusters.

**The CuTe Python DSL.** Since CUTLASS 4.0, the same CuTe concepts are available from Python: you write the kernel in Python with `@cute.kernel` and `@cute.jit`, and it is compiled to GPU code at run time, no C++ templates involved. The repository has Hopper and Blackwell kernels under `examples/python/CuTeDSL/cute/hopper` and `.../cute/blackwell`, including a small TMA tutorial and a tutorial GEMM, and there is a separate `blackwell_geforce` folder for sm_120.

The DSL itself is not limited to new GPUs; it also targets Ampere-class `mma.sync`. So we can at least run a hello world on the 4090:

{{< code file="cuda/post20/hello_cute.py" >}}

```bash
python3 -m venv venv
venv/bin/pip install nvidia-cutlass-dsl
venv/bin/python hello_cute.py
```

Output (RTX 4090, `nvidia-cutlass-dsl` 4.8.0):

```text
Hello from block 1
Hello from block 0
```

The block order is not fixed, as in post 00.

**Is the CuTe DSL still in beta?** The plan for this series said it was expected to leave beta by the end of summer 2026. As of 2026-09-29, it has not, at least not officially: the latest release, `nvidia-cutlass-dsl` 4.8.0 (published on PyPI on 2026-09-21), still carries the `Development Status :: 4 - Beta` classifier, and the CUTLASS README still says "CuTe DSL is currently in public beta and will graduate out of beta by end of summer 2026" [6][7]. So: usable, actively released, but expect API changes between versions, and pin the version you use.

---

## 9. Compile Check

We cannot run the program, but we can prove it compiles. Here is the full file, with both kernels and a host `main` that checks the compute capability first:

{{< code file="cuda/post20/main.cu" >}}

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=sm_90 -O3 main.cu -o hopper-concepts -lcuda
./hopper-concepts
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post20/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/hopper-concepts
```
{{< /tab >}}
{{< /tabs >}}

Note that this is the first post where we do **not** use `-arch=native` / `CMAKE_CUDA_ARCHITECTURES native`. On a 4090, `native` means `sm_89`, and the compiler rejects the code immediately:

```text
main.cu(40): error: __cluster_dims__ is not supported for this GPU architecture
main.cu(43): error: namespace "cooperative_groups" has no member "cluster_group"
```

Compiled for `sm_90`, it builds without warnings, and on the 4090 the guard at the top of `main` stops it politely:

{{< code file="cuda/post20/main.cu" region="guard" link="false" >}}

Output (RTX 4090):

```text
Compute capability 8.9: clusters and TMA need 9.0+. Nothing to run.
```

Without the guard, the first launch would fail with `no kernel image is available for execution on the device`, because the binary only contains `sm_90` code. On an H100 or B200, the program should print `PASSED` for both parts; we have not run it on one, so treat that as untested.

One more flag detail if you go further. For the `wgmma` and `tcgen05` snippets, use an explicit `-gencode`:

```bash
nvcc -U_GNU_SOURCE -gencode arch=compute_90a,code=sm_90a  ...   # wgmma
nvcc -U_GNU_SOURCE -gencode arch=compute_100a,code=sm_100a ...  # tcgen05
```

With CUDA 13.1, the shorter `-arch=sm_90a` also generated plain `compute_90` PTX, and `ptxas` then rejected `wgmma` for `sm_90`. In CMake, `set(CMAKE_CUDA_ARCHITECTURES 90a)` does the right thing.

---

## 10. Exercise

Pick one of these, depending on how much you want to spend.

1. **Free, on paper.** In `cluster_sum`, delete the second `cluster.sync()`. Explain what can go wrong, and why the program might still print `PASSED` most of the time (think back to the missing bounds check in post 01). Then answer: why must the grid size be a multiple of the cluster size, and what would happen to the last cluster otherwise?
2. **Free, with the compiler.** Take the `tcgen05` skeleton from section 6, put it in a file, and assemble it with `-gencode arch=compute_XX,code=sm_XX` for `90a`, `100a`, `103a`, `110a` and `120a`. Make your own version of the table in section 1 from what `ptxas` accepts.
3. **A few dollars.** Rent an H100 (or B200) by the hour from a cloud GPU provider. Run `hopper-concepts` there, then run it under `compute-sanitizer` with the second `cluster.sync()` removed. Then build `examples/48_hopper_warp_specialized_gemm` from CUTLASS and compare its TFLOP/s with `cublasGemmEx` on the same matrix sizes, the same way we compared our SGEMM with cuBLAS in post 18.

In summary, Hopper and data-center Blackwell change who does the work. Clusters let blocks cooperate and read each other's shared memory. TMA lets one thread move a whole tile, with an `mbarrier` counting the bytes. `wgmma` lets a warpgroup multiply straight out of shared memory, and `tcgen05` lets a single thread do it with the result kept in tensor memory. None of it runs on a 4090, and the RTX 50 series only gets clusters and TMA, not `wgmma` or `tcgen05`. Everything from the earlier posts, like coalescing, shared memory tiles and occupancy, is still the foundation; these features only change how the data gets to the Tensor Cores. For real work, reach them through CUTLASS or the CuTe DSL (still in beta at the time of writing) instead of raw PTX.

[1]: https://docs.nvidia.com/cuda/cuda-programming-guide/ "CUDA Programming Guide (thread block clusters, distributed shared memory, TMA)"
[2]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html "NVIDIA Hopper Tuning Guide"
[3]: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html "NVIDIA Blackwell Tuning Guide"
[4]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html "PTX ISA: cp.async.bulk.tensor, mbarrier, wgmma, tcgen05"
[5]: https://github.com/NVIDIA/cutlass "NVIDIA CUTLASS (C++ and CuTe DSL), README checked 2026-09-29"
[6]: https://pypi.org/project/nvidia-cutlass-dsl/ "nvidia-cutlass-dsl on PyPI: 4.8.0, 2026-09-21, Development Status 4 - Beta"
[7]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html "CuTe DSL introduction"
[8]: https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/ "Blackwell GPU Wiki: SM100 vs SM120 (community)"
[9]: https://gau-nernst.github.io/tcgen05/ "tcgen05 for dummies (gau-nernst)"
[10]: https://research.colfax-intl.com/tutorial-hopper-tma/ "Colfax: Mastering the NVIDIA TMA"
[11]: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/ "Colfax: CUTLASS tutorial, wgmma on Hopper"
