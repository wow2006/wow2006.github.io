---
title: "Post 17: Tensor Cores (on the 4090)"
date: 2026-09-29T19:17:00+03:00
weight: 18
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 07, we wrote a tiled matrix multiply and compared it with cuBLAS. Our best kernel reached 6.1 TFLOP/s, and cuBLAS SGEMM reached 54.3 TFLOP/s. That is already close to the 82.6 TFLOP/s the RTX 4090 can do with its regular FP32 cores. So is 82.6 the ceiling? No. Every SM of the 4090 also has *tensor cores*: small units that do nothing but multiply matrices. In post 16, we met `__half`, the 16-bit float. Tensor cores take `__half` inputs, and with them the ceiling doubles to 165.2 TFLOP/s.

In this post, we will program the tensor cores ourselves: first with the WMMA API, then one level lower with an inline PTX `mma.sync` instruction. We will check both against cuBLAS, and finish with a quick look at CUTLASS, the NVIDIA library that sits between writing everything by hand and calling cuBLAS.

## What We Will Learn

In this article, we will cover:

* What a tensor core computes, and how fast it is on the RTX 4090.
* How to write a half-precision GEMM with the WMMA API (`nvcuda::wmma`).
* How to call the `mma.sync` instruction directly with inline PTX, and how its data is spread over the 32 threads of a warp.
* How to run the same GEMM with `cublasGemmEx`.
* What CUTLASS is, and how to use it in a few lines.

## Table of Contents

1. What a Tensor Core Does.
2. The Full Program.
3. The WMMA Kernel.
4. One Level Down: `mma.sync` in Inline PTX.
5. The cuBLAS Reference.
6. Build and Run.
7. A Quick Tour of CUTLASS.
8. Exercise.

---

## 1. What a Tensor Core Does

A regular CUDA core does one fused multiply-add per clock: `d = a * b + c` on single numbers. A tensor core does the same thing on small matrices:

```text
D = A * B + C
```

The warp is the unit that talks to the tensor cores. All 32 threads of a warp issue one *matrix multiply-accumulate* (MMA) instruction together. Each thread holds a few elements of `A`, `B`, and `C` in its registers, and the hardware combines them into one tile operation. No single thread owns the whole matrix.

On the RTX 4090 (Ada, compute capability 8.9) the most common shape is `m16n8k16`: `A` is 16×16, `B` is 16×8, and `C`/`D` are 16×8. The inputs are FP16 (or BF16), and the accumulator is FP16 or FP32. One such instruction does 16 × 8 × 16 = 2048 multiply-adds, or 4096 floating-point operations.

How fast is that? From NVIDIA's Ada whitepaper [1] (Appendix A, dense numbers, at the 2520 MHz boost clock):

| RTX 4090 peak                                  | TFLOP/s |
| ---------------------------------------------- | ------- |
| FP32, regular CUDA cores                        | 82.6    |
| FP16 tensor cores, FP32 accumulate              | 165.2   |
| FP16 tensor cores, FP16 accumulate              | 330.3   |

The whitepaper also lists numbers twice as big again; those are for *sparse* matrices, a special 2:4 format. We use dense matrices, so our target is **165.2 TFLOP/s**: FP16 in, FP32 accumulate. We accumulate in FP32 because adding 4096 products in FP16 loses too much precision (we will see how much in the exercise).

Notice that FP32 accumulate is half the speed of FP16 accumulate. This is a GeForce limit. On data-center cards like the A100 both run at the same speed.

---

## 2. The Full Program

```bash
mkdir tensor-cores
cd tensor-cores
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post17/main.cu" >}}

We multiply two 4096×4096 matrices. `A` is row-major as usual. `B` is stored *column-major*: element `(k, n)` lives at `b[n * K + k]`. In other words, each column of `B` is contiguous in memory, just like each row of `A`. This is the layout the tensor core instructions want for `B`, and it keeps all three versions simple.

{{< code file="cuda/post17/main.cu" region="host" link="false" >}}

`__float2half` works on the host too, so we fill the matrices on the CPU with random values in `[-1, 1]` and round them to FP16. The output `C` is `float`.

We keep `CUDA_CHECK` and `DeviceBuffer` from post 01, but `DeviceBuffer` is now a template, because we need buffers of `half` and of `float`. `time_ms` is the cudaEvent timer from post 03.

---

## 3. The WMMA Kernel

WMMA (*warp matrix multiply-accumulate*) is the C++ API for tensor cores. It lives in `<mma.h>`, in the namespace `nvcuda::wmma`, and hides the details of which thread holds which element.

{{< code file="cuda/post17/main.cu" region="wmma" link="false" >}}

There are three new types, all called `fragment`:

* `fragment<matrix_a, 16, 16, 16, half, row_major>` is a 16×16 tile of `A`.
* `fragment<matrix_b, 16, 16, 16, half, col_major>` is a 16×16 tile of `B`.
* `fragment<accumulator, 16, 16, 16, float>` is a 16×16 tile of `C`.

A fragment is spread over the registers of the 32 threads in the warp. We do not know, and do not need to know, which element sits in which thread. We only use four functions on them:

| Function                  | What it does                                                  |
| ------------------------- | ------------------------------------------------------------- |
| `wmma::fill_fragment`     | Set every element of a fragment to a value.                   |
| `wmma::load_matrix_sync`  | Load a 16×16 tile from memory, given a pointer and the row stride. |
| `wmma::mma_sync`          | `acc = a * b + acc` on the tensor cores.                      |
| `wmma::store_matrix_sync` | Write a 16×16 tile back to memory.                            |

The `_sync` suffix means all 32 threads of the warp must call it together, with the same arguments. It is a warp-wide operation, not a per-thread one.

The rest is the same loop as in post 07, one level up. Instead of one thread computing one element, one *warp* computes a 32×32 tile of `C`, as 2×2 fragments. It walks along `K` in steps of 16: load two tiles of `A`, two tiles of `B`, and do four MMAs. Each loaded fragment is used twice, which halves the loads compared with one fragment per warp.

We launch 2×2 warps per block, so a block covers 64×64 of `C`:

{{< code file="cuda/post17/main.cu" region="launch" link="false" >}}

`threads(64, 2)` gives two warps along `x` (`threadIdx.x / 32` is the warp's column) and two along `y`. To keep the code short, the kernel assumes `M` and `N` are multiples of 64 and `K` is a multiple of 16; there are no bounds checks.

---

## 4. One Level Down: `mma.sync` in Inline PTX

WMMA is portable, but it hides the real hardware instruction. PTX is the virtual assembly language that `nvcc` compiles CUDA C++ into, and on Ampere and Ada the tensor core instruction is:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Read it from left to right: warp-synchronous (`sync`, `aligned`), shape 16×8×16, `A` is row-major and `B` is column-major (`row.col`), and the types are `D = f32`, `A = f16`, `B = f16`, `C = f32`.

We can call it from C++ with inline assembly:

{{< code file="cuda/post17/main.cu" region="mma-ptx" link="false" >}}

`asm volatile` inserts the PTX line as is. `%0` to `%9` refer to the operands after the colons. `"+f"` means a `float` register that is both read and written (our accumulator), and `"r"` means a 32-bit register that is only read. Two `half` values are packed into one 32-bit register, so the 8 halves of `A` a thread owns take 4 registers, and the 4 halves of `B` take 2. `load_pair` reads two neighbouring halves as one `uint32_t`.

This time, nothing hides the layout: each thread must load exactly the elements the hardware expects. The PTX ISA documentation [2] draws this with pictures. For `m16n8k16` it boils down to two numbers per lane:

```text
g = lane / 4    // 0..7
t = lane % 4    // 0..3
```

| Register | Holds                                                   |
| -------- | ------------------------------------------------------- |
| `a[0]`   | `A[g][2t]`,     `A[g][2t+1]`                            |
| `a[1]`   | `A[g+8][2t]`,   `A[g+8][2t+1]`                          |
| `a[2]`   | `A[g][2t+8]`,   `A[g][2t+9]`                            |
| `a[3]`   | `A[g+8][2t+8]`, `A[g+8][2t+9]`                          |
| `b[0]`   | `B[2t][g]`,     `B[2t+1][g]`                            |
| `b[1]`   | `B[2t+8][g]`,   `B[2t+9][g]`                            |
| `d[0..1]`| `D[g][2t]`,     `D[g][2t+1]`                            |
| `d[2..3]`| `D[g+8][2t]`,   `D[g+8][2t+1]`                          |

So lanes 0 to 3 share row 0 of `A`, lanes 4 to 7 share row 1, and so on. Each group of four lanes covers one row (and the row 8 below it), and each lane in the group takes two neighbouring columns. The pairs are always two consecutive `k` values. That is why we stored `B` column-major: `B[2t][g]` and `B[2t+1][g]` are then next to each other in memory, and one 32-bit load gets both.

Here is the kernel. It is the WMMA kernel with the fragments written out by hand. A 32×32 warp tile is now 2 tiles of 16 rows times 4 tiles of 8 columns:

{{< code file="cuda/post17/main.cu" region="mma-kernel" link="false" >}}

The four `load_pair` calls for `A` follow the table row by row, and so do the stores at the end: `acc[i][j][0..1]` go to row `g`, `acc[i][j][2..3]` to row `g + 8`.

Both kernels end up as the same machine instruction. We can check with `cuobjdump`, which prints the SASS (the real GPU assembly) inside a binary:

```bash
cuobjdump -sass tensor-cores | grep -oE "HMMA\.[0-9A-Z.]+" | sort | uniq -c
```

```text
     64 HMMA.16816.F32
```

`HMMA.16816.F32` is `m16n8k16` with FP32 accumulate. A WMMA 16×16×16 step is simply two of them side by side.

A real kernel would not load fragments straight from global memory like we do. It would first copy tiles into shared memory, then use the `ldmatrix` instruction, which loads a whole fragment from shared memory in exactly this layout in one go. We skip that here to keep the focus on the MMA itself.

---

## 5. The cuBLAS Reference

We compare both kernels with cuBLAS. For mixed precision, the function is `cublasGemmEx`, which takes a data type for each matrix and a separate *compute type*:

{{< code file="cuda/post17/main.cu" region="cublas" link="false" >}}

`CUDA_R_16F` says `A` and `B` are FP16, `CUDA_R_32F` says `C` is FP32, and `CUBLAS_COMPUTE_32F` says to accumulate in FP32. cuBLAS picks tensor cores on its own.

cuBLAS is column-major, like Fortran. We used the same trick in post 07: our row-major `C` is, in cuBLAS's eyes, a column-major `C^T`, and `C^T = B^T A^T`. So we pass `B` first, then `A`. Our column-major `B` looks like a row-major `B^T` to cuBLAS, which is why we ask it to transpose that one (`CUBLAS_OP_T`). No data is copied.

We also need to check that we called cuBLAS correctly. So before we trust it as the reference, we compare 256 random elements against a CPU sum in `double`:

{{< code file="cuda/post17/main.cu" region="verify" link="false" >}}

Then we run each version, compare its whole output with cuBLAS, and time it:

{{< code file="cuda/post17/main.cu" region="measure" link="false" >}}

We count `2 * M * N * K` floating-point operations: one multiply and one add for each of the `K` terms of each of the `M * N` outputs.

---

## 6. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o tensor-cores -lcublas
./tensor-cores
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post17/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/tensor-cores
```
{{< /tab >}}
{{< /tabs >}}

We need `-arch=native` this time. Without it, `nvcc` compiles for its default architecture, which is older than Ampere, and the build fails:

```text
ptxas /tmp/tmpxft_0004968d_00000000-6_main.ptx, line 457; error   : Feature '.m16n8k16' requires .target sm_80 or higher
```

`mma.sync` with the `m16n8k16` shape needs at least compute capability 8.0 (Ampere). WMMA alone would still compile for older GPUs, because it can pick an older instruction.

Output:

```text
cuBLAS vs CPU (256 samples): max abs error 1.88e-04
M = N = K = 4096, FP16 inputs, FP32 accumulate
cuBLAS      0.903 ms   152.2 TFLOP/s   92.1% of peak  max abs diff 0.00e+00
WMMA        3.105 ms    44.3 TFLOP/s   26.8% of peak  max abs diff 8.39e-04
mma.sync    3.232 ms    42.5 TFLOP/s   25.7% of peak  max abs diff 8.39e-04
PASSED
```

The results are around 20 in size, so an error below `1e-3` is just FP32 rounding in a different order. cuBLAS against itself is exactly 0, as expected.

A few things stand out:

* **cuBLAS reaches 152 TFLOP/s, 92% of the tensor peak.** In post 07, cuBLAS SGEMM did 54.3 TFLOP/s on the same size. Same library, same card, 2.8 times faster, just by switching the inputs to FP16 and letting the tensor cores work.
* **Our two kernels reach about 43 TFLOP/s.** That is about 7 times our best FP32 kernel from post 07, with less code. Yet it is only a quarter of the peak.
* **WMMA and `mma.sync` are about equal.** They compile to the same `HMMA` instruction, so the choice between them does not change the math. What `mma.sync` gives us is control over the layout, which we need for the next steps.

Why only a quarter? A tensor core finishes an MMA so fast that the real work is feeding it. Our kernels load every fragment straight from global memory (through L1 and L2), and each value is reused only twice. cuBLAS stages big tiles in shared memory, loads them with `ldmatrix`, overlaps the next copy with the current math, and reuses each value many more times. Those are exactly the steps we will build one by one in the capstone, post 18.

Note that 92% is measured against 165.2 TFLOP/s, which the whitepaper computes at the 2520 MHz boost clock. A 4090 often runs a bit faster than that under load, so a library can land very close to, or even slightly above, "100%".

We tried to profile this with `ncu`, but on this machine it fails with `ERR_NVGPUCTRPERM`: the driver only lets root read the performance counters. Run it with `sudo`, or allow all users by loading the `nvidia` kernel module with `NVreg_RestrictProfilingToAdminUsers=0`.

---

## 7. A Quick Tour of CUTLASS

Between "call cuBLAS" and "write `mma.sync` by hand", there is CUTLASS [3]. It is NVIDIA's open-source C++ template library for GEMM and convolution. It is header-only: you pick tile sizes, data types, and layouts as template parameters, and the compiler builds the kernel for you. CUTLASS is how you get cuBLAS-level speed while still being able to change things, for example to fuse an activation function into the end of the GEMM. It also contains CuTe, the layout library its newer kernels are built on, which we will meet again in post 20.

Here is the same GEMM with CUTLASS:

{{< code file="cuda/post17/cutlass/main.cu" region="gemm" link="false" >}}

The first seven parameters describe the problem: the type and layout of `A`, `B`, and `C`, the accumulator type, and `OpClassTensorOp` for tensor cores. `Sm80` selects the Ampere code path, which is the one that uses `mma.sync`; it runs on the 4090 too. The last three are the tiles: each block computes 128×128 of `C` and walks `K` in steps of 32, each warp computes 64×64, and the smallest unit is our `m16n8k16` instruction. We set the block tile ourselves because the default for `Sm80` (128×256×64, 3 stages) needs about 144 KB of shared memory, and the 4090 only allows 99 KB per block.

Running it looks like this:

{{< code file="cuda/post17/cutlass/main.cu" region="run" link="false" >}}

Each matrix is passed as a pointer plus its leading dimension. `C` is the input that gets scaled by `beta`, and `D` is the output; here they are the same buffer.

The full program, [`cutlass/main.cu`](https://github.com/wow2006/wow2006.github.io/blob/main/assets/code/cuda/post17/cutlass/main.cu), fills the matrices the same way as before, checks 256 elements against the CPU, and times the GEMM. CMake can download CUTLASS for us with `FetchContent`:

{{< code file="cuda/post17/cutlass/CMakeLists.txt" >}}

`SOURCE_SUBDIR do-not-build` points `FetchContent` at a folder that does not exist, so it downloads CUTLASS but does not try to build its large CMake project; we only need the headers. `--expt-relaxed-constexpr` silences warnings from CUTLASS headers, and CUTLASS's own build adds the same flag.

CUTLASS pulls in `<mutex>`, which hits the same `-U_GNU_SOURCE` problem as Thrust and CUB did in post 11. So the file starts with the same small shim:

{{< code file="cuda/post17/cutlass/main.cu" region="shim" link="false" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/cutlass-gemm
```

Output:

```text
CUTLASS     0.852 ms   161.4 TFLOP/s   97.7% of peak  max abs error vs CPU 6.56e-04
```

161 TFLOP/s: a little faster than cuBLAS on this size, from about ten lines of template parameters. Across three runs, CUTLASS was consistently between 160.7 and 161.8, and cuBLAS at 152. This does not mean CUTLASS is always faster. cuBLAS has to pick a good kernel for every shape without being told anything, while we picked our tiles for one shape and one GPU.

---

## 8. Exercise

We accumulated in FP32 because the whitepaper halves the speed for it on GeForce cards. Let's see what we would win and lose with FP16 accumulate.

Change the cuBLAS call to accumulate in FP16: make `C` a `half` buffer, use `CUDA_R_16F` for it, use `CUBLAS_COMPUTE_16F`, and pass `alpha` and `beta` as `half` values (the scalars must match the compute type). Then check a few elements against the CPU, like in section 5.

1. How close do you get to the 330.3 TFLOP/s peak?
2. How does the error compare with the FP32 accumulate version?

On our 4090, cuBLAS reached about 250 TFLOP/s, but the maximum error on the 256 samples grew from `1.9e-4` to `0.33`, more than a thousand times bigger. Adding 4096 terms into a number with only 11 bits of precision (post 16) throws away the low bits of every step. Faster is not free.

For a bigger challenge, change `mma_16x8x16` to the `.f16.f16.f16.f16` variant. The PTX ISA documentation [2] shows that the accumulator then shrinks from four `float` registers to two `half2` registers.

In summary, a tensor core computes `D = A * B + C` on a small tile, and a whole warp drives it together, each thread holding a piece of each matrix in its registers. WMMA gives us that as fragments without exposing the layout; `mma.sync` gives us the raw instruction, and we have to place each element in the right lane ourselves. Both reached about 43 TFLOP/s on a 4096³ half-precision GEMM, while cuBLAS reached 152 and CUTLASS 161, out of a 165.2 TFLOP/s peak. The gap is not the math, which is identical; it is how well the kernel feeds the tensor cores. Closing that gap is the subject of the next post.

[1]: https://www.nvidia.com/content/PDF/nvidia-ada-gpu-architecture.pdf "NVIDIA Ada GPU Architecture whitepaper (Appendix A: GeForce RTX 4090 Full Specifications)"
[2]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma "PTX ISA: Warp-level matrix multiply-accumulate instructions"
[3]: https://github.com/NVIDIA/cutlass "CUTLASS on GitHub"
[4]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions "CUDA Programming Guide: Warp Matrix Functions"
[5]: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex "cuBLAS: cublasGemmEx"
