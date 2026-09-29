---
title: "Post 07: Tiled matrix multiply"
date: 2026-09-29T19:07:00+03:00
weight: 8
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the previous post, we used shared memory to speed up a matrix transpose. A block loaded a tile into shared memory, waited at `__syncthreads`, and then read it back in a different order. In this post, we use the same idea on the most famous kernel of all: matrix multiplication.

Matrix multiply (called *GEMM*, or *SGEMM* for single-precision floats) is at the heart of deep learning, physics simulations, and a lot of scientific code. It is also the kernel that NVIDIA spends the most effort on. So it is a good test: we will write two versions ourselves, then measure them against cuBLAS, NVIDIA's own library, and against the peak of the RTX 4090.

Spoiler: we will not win. But we will see exactly how far behind we are, and that gap is what post 18 is about.

## What We Will Learn

In this article, we will cover:

* How to write a naive matrix multiply with one thread per output element.
* How to split the work into tiles and share them through shared memory.
* How to call `cublasSgemm` on row-major data (cuBLAS is column-major).
* How to measure GFLOP/s and check our result against cuBLAS with a tolerance.

## Table of Contents

1. Matrix Multiply in One Minute.
2. The Full Program.
3. The Naive Kernel.
4. Tiling with Shared Memory.
5. cuBLAS and the Column-Major Trick.
6. Timing and Checking.
7. Build and Run.
8. Why Is cuBLAS Still 8x Faster?
9. Exercise.

---

## 1. Matrix Multiply in One Minute

We multiply two square matrices `A` and `B` of size `n × n` and store the result in `C`. Each element of `C` is the dot product of one row of `A` and one column of `B`:

```text
C[row][col] = A[row][0] * B[0][col] + A[row][1] * B[1][col] + ... + A[row][n-1] * B[n-1][col]
```

We store the matrices *row-major*, like a C array: element `(row, col)` lives at index `row * n + col`.

How much work is that? `C` has `n × n` elements, and each one needs `n` multiplications and `n` additions. So the total is `2 · n³` floating-point operations (FLOPs). We use `n = 4096`:

```text
2 × 4096³ = 137,438,953,472 FLOPs ≈ 137 GFLOP
```

Dividing that by the kernel time gives us GFLOP/s, our score for this post. For reference, the RTX 4090 has 16,384 CUDA cores, each able to do one fused multiply-add (2 FLOPs) per clock at up to 2.52 GHz. That is about **82.6 TFLOP/s** of FP32 on paper [1].

---

## 2. The Full Program

```bash
mkdir tiled-matmul
cd tiled-matmul
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post07/main.cu" >}}

The program runs three versions of the same multiply (our naive kernel, our tiled kernel, and cuBLAS), times each one, and compares our two results with cuBLAS. Let's go through the parts.

---

## 3. The Naive Kernel

The most direct way to parallelize matrix multiply is to give each thread one element of `C`:

{{< code file="cuda/post07/main.cu" region="naive" link="false" >}}

This is the 2D indexing from post 02: `x` picks the column and `y` picks the row. Each thread then runs the dot-product loop over `k` on its own.

We launch a 2D grid of 32 × 32 blocks, enough to cover the whole `4096 × 4096` output:

{{< code file="cuda/post07/main.cu" region="launch" link="false" >}}

The choice of `x` for the column is not random. Remember from post 05 that the 32 threads of a warp should read neighbouring addresses. In a warp, `threadIdx.x` goes from 0 to 31 while `threadIdx.y` stays the same. So all 32 threads share the same `row` and have 32 consecutive `col` values. In the loop:

* `B[k * n + col]` reads 32 consecutive floats: a nicely coalesced load.
* `A[row * n + k]` is the same address for all 32 threads, so the hardware reads it once and broadcasts it.

Now count the memory traffic. Every thread reads `2n` floats, and there are `n²` threads, so the kernel asks for `2n³` floats. That is about 550 GB for one multiply, although the three matrices together are only 192 MB. The same rows of `A` and columns of `B` are read over and over again, 4096 times each.

---

## 4. Tiling with Shared Memory

The fix for "reading the same data over and over" is the same as in the transpose: load a piece once into shared memory, and let the whole block reuse it.

Look at one 32 × 32 block of `C`. To compute it, we need 32 full rows of `A` and 32 full columns of `B`. We cut those into 32 × 32 tiles and walk along them in steps:

```text
             B
          +-----+
          | B0  |   step 0: As = A0, Bs = B0
          +-----+   step 1: As = A1, Bs = B1
          | B1  |   ...
          +-----+
          | ... |
          +-----+
+----+----+-----+   +-----+
| A0 | A1 | ... |   |  C  |   C tile = A0*B0 + A1*B1 + ...
+----+----+-----+   +-----+
       A
```

At each step, the 1024 threads of the block load one tile of `A` and one tile of `B` into shared memory together, one element each. Then every thread does 32 multiply-adds using only shared memory:

{{< code file="cuda/post07/main.cu" region="tiled" link="false" >}}

The two `__syncthreads` calls are the ones we met in the previous post, and both are needed:

* The first one waits until the tiles are fully loaded. Without it, a thread could start reading `As[ty][k]` before the thread responsible for that element has written it.
* The second one waits until everyone has finished using the tiles. Without it, a fast thread could jump to the next step and overwrite `As` while a slow thread is still reading the old values.

Each element that we load from global memory is now used by 32 threads instead of one. So the kernel asks global memory for `2n³ / 32` floats instead of `2n³`, 32 times less traffic.

To keep the code short, the kernel assumes `n` is a multiple of `TILE`, so there are no bounds checks. We make the compiler enforce it on the host side:

{{< code file="cuda/post07/main.cu" region="host" link="false" >}}

The inputs are random numbers between -1 and 1, with a fixed seed so every run gets the same matrices.

---

## 5. cuBLAS and the Column-Major Trick

cuBLAS is NVIDIA's implementation of BLAS, the standard linear algebra interface. BLAS comes from Fortran, so cuBLAS expects matrices in *column-major* order: element `(row, col)` lives at `col * n + row`. Our matrices are row-major.

We could transpose everything, but there is a cheaper trick. If you read a row-major matrix as if it were column-major, you get its transpose. So cuBLAS does not see `A`, `B`, and `C`. It sees `Aᵀ`, `Bᵀ`, and `Cᵀ`.

Now we use a basic rule of linear algebra:

```text
C = A · B   <=>   Cᵀ = Bᵀ · Aᵀ
```

So we ask cuBLAS to compute `Bᵀ · Aᵀ`, which just means passing `B` as the first matrix and `A` as the second. cuBLAS writes `Cᵀ` in column-major order, which is exactly `C` in row-major order. No copies, no transposes:

{{< code file="cuda/post07/main.cu" region="cublas" link="false" >}}

`cublasSgemm` computes `C = alpha · op(A) · op(B) + beta · C`. With `alpha = 1`, `beta = 0`, and `CUBLAS_OP_N` (no transpose), that is a plain `C = A · B`. The three `n` arguments are the sizes `m`, `n`, and `k`, and the other three `n` are the *leading dimensions*, the distance in floats between two columns. For a dense square matrix, all of them are `n`.

Every cuBLAS call needs a *handle*, which holds the library state. We create it with `cublasCreate` and destroy it with `cublasDestroy`. Like `cudaFree` in post 01, `cublasDestroy` fits perfectly as a `unique_ptr` deleter. `cublasHandle_t` is just a pointer to a `cublasContext`, so the handle is released on every path out of `main`.

cuBLAS functions do not return a `cudaError_t` but a `cublasStatus_t`, so our `CUDA_CHECK` macro does not apply. We check it against `CUBLAS_STATUS_SUCCESS` by hand.

---

## 6. Timing and Checking

We time each version the way we learned in post 03: one warm-up launch, then ten launches between two CUDA events, then the average:

{{< code file="cuda/post07/main.cu" region="timer" link="false" >}}

The timer takes a lambda, so we can use it for our kernels and for the cuBLAS call alike.

For the check, we run cuBLAS once first and keep its result as the reference. After each of our kernels, we copy `C` back and find the largest difference from the reference:

{{< code file="cuda/post07/main.cu" region="verify" link="false" >}}

Why not compare with `==` like in post 01? Because floating-point addition is not associative: `(a + b) + c` is not always exactly `a + (b + c)`. Our kernels add the 4096 products in order, from `k = 0` to `k = 4095`. cuBLAS splits the sum into pieces and adds them in a different order. Both answers are correct, but the last bits differ. The elements of `C` here are around ±20, so an error of `1e-3` would still mean about five correct significant digits. A real bug, like a wrong index or a missing `__syncthreads`, gives errors many orders of magnitude larger.

Finally, we print one line per version:

{{< code file="cuda/post07/main.cu" region="report" link="false" >}}

---

## 7. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o tiled-matmul -lcublas
./tiled-matmul
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post07/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/tiled-matmul
```
{{< /tab >}}
{{< /tabs >}}

cuBLAS is a separate library, so we need to link it. With `nvcc`, that is `-lcublas` at the end. With CMake, `find_package(CUDAToolkit)` finds the toolkit libraries and gives us the imported target `CUDA::cublas`.

Output: (RTX 4090)

```text
SGEMM 4096 x 4096 x 4096, 10 runs each

kernel     time ms    GFLOP/s   % peak   % cuBLAS    max err
naive        26.71       5145     6.2%       9.5%   3.89e-04
tiled        22.41       6134     7.4%      11.3%   3.89e-04
cuBLAS        2.53      54274    65.7%     100.0%   0.00e+00

PASSED (tolerance 1e-03)
```

Both of our kernels agree with cuBLAS to within `4e-4`. They have exactly the same error because they add the products in the same order, so they produce the same bits.

Your numbers will move by a few percent between runs, but the picture stays the same:

* The tiled kernel is only about **1.2x** faster than the naive one.
* cuBLAS is about **8.8x** faster than our tiled kernel.
* Even cuBLAS reaches only about two-thirds of the 82.6 TFLOP/s paper peak.

---

## 8. Why Is cuBLAS Still 8x Faster?

The small gain from tiling is surprising at first. We cut global memory requests by 32x and only got 20% faster. Why?

**The naive kernel was not as naive as it looked.** It asks for 550 GB in 26.7 ms. That is about 20 TB/s, twenty times the ~1 TB/s the 4090's memory can deliver. That is impossible from DRAM alone, so most of those reads never reach it. The 4090 has a large L1 cache in every SM and a 72 MB L2 cache, and they catch most of the repeated reads for us. The broadcast of `A` and the coalesced reads of `B` also help. On older GPUs with smaller caches, the difference between naive and tiled is much larger.

**The tiled kernel has a new bottleneck: shared memory.** Look at the inner loop again:

```cpp
sum += As[ty][k] * Bs[k][tx];
```

For each multiply-add, a thread does two shared-memory reads. Shared memory is fast, but not fast enough to feed the math units at one fused multiply-add per two loads. The math units spend most of their time waiting for data.

**What cuBLAS does differently.** The key idea is to make each thread compute many outputs instead of one. If a thread computes an 8 × 8 patch of `C`, it can load 8 values of `A` and 8 values of `B` into registers and do 64 multiply-adds with them. That is 16 loads for 64 multiply-adds instead of 2 loads for 1. This is called *register tiling*, and together with wider vector loads, copying the next tile while computing the current one, and careful tuning, it gets close to the numbers above.

These optimizations are too much for one post. In post 18, we will add them one at a time, measure each step, and watch our kernel climb toward cuBLAS.

One more note. The 4090 also has Tensor Cores, which can multiply matrices much faster using lower precision. `cublasSgemm` does not use them by default, so the comparison above is a fair FP32 against FP32. We will meet Tensor Cores in post 17.

---

## 9. Exercise

Change the tile size to 16:

```cpp
constexpr int TILE = 16;
```

Since the launch uses `TILE` too, both kernels now run with 16 × 16 = 256 threads per block. Build and run again. On our RTX 4090, it looks like this:

```text
kernel     time ms    GFLOP/s   % peak   % cuBLAS    max err
naive        27.05       5081     6.2%       9.5%   3.89e-04
tiled        20.07       6848     8.3%      12.8%   3.89e-04
cuBLAS        2.57      53484    64.8%     100.0%   0.00e+00
```

The naive kernel does not care, but the tiled one gets about 10% faster, even though each loaded element is now reused only 16 times instead of 32. Now try `TILE = 8`. On our machine, the tiled kernel drops to about 5,300 GFLOP/s, slower than both 16 and 32. So the middle size wins. Can you explain why? Hint: a small tile means less reuse of each loaded element; a big tile means 1024 threads all waiting for each other at every `__syncthreads`, and fewer blocks living on one SM at the same time. We come back to this in post 08 on occupancy.

For a harder challenge, remove the `static_assert` and make `matmul_tiled` work for any `n`, for example `n = 4000`. Threads that fall outside the matrix should load `0.0f` into the tile instead of reading out of bounds, and should skip the final write. They still must reach both `__syncthreads`. Check your version with `compute-sanitizer`.

In summary, the naive matrix multiply gives each thread one element of `C` and reads its inputs straight from global memory. Tiling lets a block load each piece of `A` and `B` into shared memory once and reuse it many times, with two `__syncthreads` to keep the loads and reads in order. cuBLAS works on column-major data, but passing `B` before `A` gives us a row-major result for free. On the RTX 4090, caches already make the naive kernel decent, tiling adds about 20%, and cuBLAS is still almost 9x ahead. Closing that gap is the job of post 18.

[1]: https://www.nvidia.com/content/PDF/nvidia-ada-gpu-architecture.pdf "NVIDIA Ada GPU Architecture whitepaper"
[2]: https://docs.nvidia.com/cuda/cublas/ "cuBLAS documentation"
[3]: https://docs.nvidia.com/cuda/cuda-programming-guide/ "CUDA Programming Guide"
[4]: https://siboehm.com/articles/22/CUDA-MMM "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"
