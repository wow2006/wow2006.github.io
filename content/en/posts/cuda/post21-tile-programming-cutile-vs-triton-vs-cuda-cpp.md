---
title: "Post 21: Tile programming in 2026: cuTile vs Triton vs CUDA C++"
date: 2026-09-29T19:21:00+03:00
weight: 22
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

For twenty posts, we have programmed the GPU one thread at a time. Each thread computed its own index, loaded its own element, and we arranged the threads into warps and blocks so that their memory accesses lined up. In post 17, we saw how much work it takes to feed the tensor cores by hand, and in post 18 how many small steps it takes to get a matrix multiply close to cuBLAS.

There is another way to write kernels. Instead of describing what *one thread* does, we describe what *one block* does to a *tile* of data: "load this 128×32 tile of A, multiply it by that tile of B, add it to the accumulator." The compiler then decides how the threads split that work, which instructions to use, and how to move the data. This is called *tile programming*. Triton made it popular, and in December 2025 NVIDIA shipped its own version, CUDA Tile, with CUDA 13.1.

In this last post of the series, we will write the same two kernels three ways, in CUDA C++, Triton, and cuTile Python, and run all of them on the RTX 4090.

## What We Will Learn

In this article, we will cover:

* What tile programming is, and how it differs from the thread-level CUDA we have written so far.
* Where cuTile stands in September 2026: languages, GPUs, and what `tileiras` is.
* A vector add and an FP16 matrix multiply in CUDA C++, Triton, and cuTile.
* How fast each version is on the RTX 4090.
* When to pick which.

## Table of Contents

1. Threads vs Tiles.
2. cuTile in 2026.
3. The CUDA C++ Version.
4. The Triton Version.
5. The cuTile Version.
6. Build and Run.
7. Comparing the Results.
8. When to Pick Which.
9. Exercise.
10. Where to Go From Here.

---

## 1. Threads vs Tiles

Here is the vector add from post 01 again:

{{< code file="cuda/post21/main.cu" region="add" link="false" >}}

Everything is written from the point of view of one thread. We compute `i`, we check the bound, we add one pair of numbers. How the threads are grouped, how many there are per block, and how their loads combine into wide memory transactions is our problem.

In a tile language, the unit of work is a block (Triton calls it a *program*), and the values we work with are whole arrays. We say "load 1024 elements starting here, add them, store them". There is no `threadIdx`. The compiler still turns this into threads and warps, of course, but it chooses how.

This matters most for kernels like matrix multiply. In post 17, we had to know that a WMMA fragment is 16×16, which lanes hold which elements for `mma.sync`, and how to split a 64×64 block tile across four warps. In a tile language, a matrix multiply of two tiles is one function call, and the compiler picks the tensor core instructions, the shared memory layout, and the pipelining for us.

The price is control. We cannot say exactly which thread does what, and when the compiler makes a poor choice, it is harder to fix.

---

## 2. cuTile in 2026

CUDA Tile has two parts [1]:

* **CUDA Tile IR**, a new intermediate representation for tile programs, next to PTX (the one for thread programs).
* **cuTile**, the language we write. It started as a Python DSL, cuTile Python.

Things have moved fast since the first release, and many articles written in early 2026 are already out of date. Here is the timeline, from NVIDIA's own release notes and blog posts:

| Date | Release | What changed for CUDA Tile |
| ---- | ------- | -------------------------- |
| Dec 2025 | CUDA 13.1 | First release. cuTile Python only. Blackwell only (compute capability 10.x and 12.x) [1] |
| Mar 2026 | CUDA 13.2 | Ampere and Ada (8.x) supported, plus 11.x [2][3] |
| May 2026 | CUDA 13.3 | Hopper (9.0) supported. CUDA Tile for C++ released [4][5] |
| Sep 2026 | CUDA 13.4 | Tile IR 13.4, developer preview for sm_107 [3] |

So in September 2026:

* **Status**: cuTile Python is a normal, released package on PyPI (`cuda-tile`, version 1.6.0 on 2026-09-09) [6]. Some individual features are still marked as preview and may change [7].
* **Languages**: Python and, since CUDA 13.3, C++ (compile with `nvcc -std=c++20 --enable-tile`) [5].
* **GPUs**: compute capability 8.x, 9.x, 10.x, 11.x and 12.x [8]. That is Ampere, Ada, Hopper and Blackwell. The RTX 4090 (8.9) is included.

### What is `tileiras`?

Our machine has CUDA 13.1 installed, and there is a new program in `/usr/local/cuda/bin`:

```bash
tileiras --help
```

```text
OVERVIEW: tileiras: NVIDIA (R) Cuda Tile IR optimizing assembler

USAGE: tileiras [options] <tile bytecode file>
...
  --gpu-name=<value>   - Specify name of NVIDIA GPU to generate code for.
    =sm_100            -   SM 100
    =sm_103            -   SM 103
    =sm_110            -   SM 110
    =sm_120            -   SM 120
    =sm_121            -   SM 121
```

`tileiras` is to Tile IR what `ptxas` is to PTX: it takes the Tile IR bytecode that cuTile produces and turns it into machine code for one GPU. You never call it yourself; cuTile calls it the first time a kernel is launched.

Look at the list of GPUs. The `tileiras` from CUDA 13.1 only knows Blackwell, so it cannot compile anything for our `sm_89`. We do not need to upgrade the whole toolkit to fix this. The `cuda-tile` Python package can bring its own, newer `tileiras`:

```bash
pip install cuda-tile[tileiras]
```

That is what we will do in section 6.

---

## 3. The CUDA C++ Version

We need nothing new here. The vector add is the one from post 01, and the matrix multiply is the WMMA kernel from post 17, with one change: B is now stored row-major (K×N), like the Python versions will use. We compare it against cuBLAS, as in post 17, and time everything with `time_ms` from post 03.

{{< code file="cuda/post21/main.cu" >}}

The WMMA kernel is the part to look at:

{{< code file="cuda/post21/main.cu" region="wmma" link="false" >}}

Notice how much of it is about *where* things are: `warp_row`, `warp_col`, the 2×2 fragments per warp, the leading dimensions. The actual math is one line, `mma_sync`. And this is still the simple version. It loads every fragment straight from global memory, with no shared memory tiles and no double buffering; post 18 showed how long the list of such steps gets.

---

## 4. The Triton Version

Triton [9] is an open-source language and compiler from OpenAI, now developed in the open by many companies. You write kernels in Python, decorated with `@triton.jit`, and they run on NVIDIA GPUs with compute capability 8.0 and newer (and on AMD GPUs). It is what PyTorch's `torch.compile` generates for the GPU, so a lot of kernels in the machine learning world are already Triton kernels.

{{< code file="cuda/post21/triton_kernels.py" >}}

### Vector add

{{< code file="cuda/post21/triton_kernels.py" region="add" link="false" >}}

`tl.program_id(0)` plays the role of `blockIdx.x`. There is no `threadIdx`: `tl.arange(0, BLOCK)` is a whole vector of 1024 offsets, and `tl.load` loads all of them at once. The bound check becomes a *mask*: lanes where `mask` is false are not loaded or stored.

`BLOCK` is marked `tl.constexpr`, so it is a compile-time constant, like a C++ template parameter. Triton compiles a separate kernel for each value.

### Matrix multiply

{{< code file="cuda/post21/triton_kernels.py" region="matmul" link="false" >}}

Each program computes one `BM × BN` tile of C. `rows[:, None] * K + ks[None, :]` uses NumPy-style broadcasting to build a 2D tile of pointers into A. Inside the loop, `tl.dot` multiplies a `BM × BK` tile by a `BK × BN` tile. That single call is where the tensor cores, the shared memory staging, and the register layout all happen. We never wrote a fragment.

We launch it with a 2D grid, one program per output tile:

{{< code file="cuda/post21/triton_kernels.py" region="run-matmul" link="false" >}}

`num_warps=4` says how many warps each program gets. `num_stages=3` asks Triton to pipeline the loop, loading the next tiles of A and B while the tensor cores work on the current ones. This is the double buffering (with `cp.async`) that we would write by hand in C++. Here it is one argument.

---

## 5. The cuTile Version

{{< code file="cuda/post21/cutile_kernels.py" >}}

### Vector add

{{< code file="cuda/post21/cutile_kernels.py" region="add" link="false" >}}

cuTile goes one step further than Triton: there are no pointers at all. `ct.load(a, index=(pid,), shape=(TILE,))` means "view `a` as a row of tiles of size `TILE`, and load tile number `pid`". The array knows its own shape, so the tile index is all we give it.

### Matrix multiply

{{< code file="cuda/post21/cutile_kernels.py" region="matmul" link="false" >}}

This is the whole matrix multiply. `ct.bid(0)` and `ct.bid(1)` are the block's tile coordinates in C. `ct.num_tiles` tells us how many `BK`-wide tiles fit along K. `ct.load(A, index=(row, k), shape=(BM, BK))` loads the `(row, k)`-th tile of A, and `ct.mma` multiplies two tiles and adds the result to the accumulator. There is no `num_warps` and no `num_stages`: in cuTile, even the number of threads per block is chosen by the compiler.

Tile sizes must be compile-time constants, and each dimension of a tile must be a power of two [10]. `ct.Constant[int]` marks a parameter as a constant; like in Triton, a new version of the kernel is compiled for each value.

We launch it with `ct.launch`, giving it a stream, the grid, the kernel, and the arguments:

{{< code file="cuda/post21/cutile_kernels.py" region="run-matmul" link="false" >}}

cuTile works with any array that supports the standard DLPack or CUDA array interface, so PyTorch tensors, CuPy arrays and others can be passed in directly. We use PyTorch here because that is what Triton kernels are usually called with, so one environment serves both.

---

## 6. Build and Run

The CUDA C++ version builds like every other post:

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -lcublas -o tile-compare
./tile-compare
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post21/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/tile-compare
```
{{< /tab >}}
{{< /tabs >}}

For the Python versions, we create a virtual environment and install PyTorch (which brings Triton with it) and cuTile with its own `tileiras`:

```bash
python3 -m venv venv
./venv/bin/pip install torch "cuda-tile[tileiras]"
./venv/bin/python triton_kernels.py
./venv/bin/python cutile_kernels.py
```

This is a large download (a few GB, mostly PyTorch's CUDA libraries). cuTile itself needs NVIDIA driver r580 or newer [8]. We hit two snags while setting this up, and you may hit them too:

* **Triton needs the Python headers.** The first time Triton runs, it compiles a small C helper with `gcc`, and that needs `Python.h`. Our system Python 3.14 had no headers installed, and the first run failed with `CalledProcessError` from `gcc`. The fix is `sudo apt install python3-dev`, or a Python that ships its headers (we used a standalone Python 3.11).
* **pip may pick an old cuTile.** PyTorch 2.14 pins its own version of the CUDA libraries, and to satisfy that pin, pip quietly installed `cuda-tile` 1.3.0 instead of the current 1.6.0. It still ran, but slower. Check with `pip list` and, if needed, ask for the version explicitly: `pip install "cuda-tile[tileiras]==1.6.0"`. pip then warns that PyTorch wanted a different `cuda-toolkit` package; both scripts ran fine anyway.

The `tileiras` that comes with the package is newer than the one in our CUDA 13.1 install, and its list of GPUs is much longer:

```bash
./venv/lib/python3.11/site-packages/nvidia/cu13/bin/tileiras --help
```

```text
  --gpu-name=<value>             - Specify name of NVIDIA GPU to generate code for.
    =sm_80                       -   SM 80
    =sm_86                       -   SM 86
    =sm_87                       -   SM 87
    =sm_88                       -   SM 88
    =sm_89                       -   SM 89
    =sm_90                       -   SM 90
    =sm_100                      -   SM 100
    =sm_103                      -   SM 103
    =sm_107                      -   SM 107
    =sm_110                      -   SM 110
    =sm_120                      -   SM 120
    =sm_121                      -   SM 121
```

There is `sm_89`.

Output of the CUDA C++ version:

```text
vector add   n = 67108864    0.864 ms   932.5 GB/s  errors 0
matmul 4096x4096x4096, FP16 in, FP32 out
  cuBLAS       0.894 ms   153.7 TFLOP/s  max abs err vs CPU    1.59e-04
  WMMA         1.974 ms    69.6 TFLOP/s  max abs diff vs cuBLAS 8.39e-04
PASSED
```

Output of the Triton version:

```text
vector add   n = 67108864    0.871 ms   924.5 GB/s  match
matmul 4096x4096x4096, FP16 in, FP32 out
  Triton       0.849 ms   161.9 TFLOP/s  max abs diff vs FP32 9.84e-04
PASSED
```

Output of the cuTile version:

```text
vector add   n = 67108864    0.871 ms   924.5 GB/s  match
matmul 4096x4096x4096, FP16 in, FP32 out
  cuTile       1.022 ms   134.5 TFLOP/s  max abs diff vs FP32 9.77e-04
PASSED
```

(PyTorch also prints a warning if NumPy is not installed. It is harmless; `pip install numpy` silences it.)

The matrix multiply results are checked against a plain FP32 matrix multiply, so a difference around `1e-3` is the normal rounding of an FP32 accumulator over 4096 products, not a bug.

---

## 7. Comparing the Results

We ran all three programs three times. The numbers were stable, except cuBLAS, which switched between two speeds from run to run:

| Kernel | CUDA C++ | Triton | cuTile |
| ------ | -------- | ------ | ------ |
| Vector add (GB/s) | 932 | 925 | 925 |
| Matrix multiply, hand-written (TFLOP/s) | 70 to 72 (WMMA) | 162 | 135 |
| Matrix multiply, cuBLAS (TFLOP/s) | 154 to 160 | | |
| Lines in the matmul kernel | 27 | 14 | 9 |

A few things stand out.

**The vector add is the same everywhere.** All three reach about 925 to 932 GB/s, the same 92% of the 1008 GB/s peak that we measured in post 03. A vector add has nothing to optimize except memory access, and all three compilers get that right. The small gap is real (the difference was the same on every run), but it is under 1%. For memory-bound kernels, the language does not matter much.

**The matrix multiply is where tiles pay off.** Our WMMA kernel from post 17 reaches about 70 TFLOP/s. The Triton kernel, which is half as long and never mentions a tensor core, reaches 162 TFLOP/s: as fast as cuBLAS on this size, and 98% of the 165.2 TFLOP/s peak we used in post 17. The Triton compiler did the work of post 18 for us: shared memory staging, a software pipeline (`num_stages=3`), and a good register layout.

(The WMMA kernel is faster here than the 44 TFLOP/s it reached in post 17. The only change is that B is row-major instead of column-major. It still has no shared memory, so it is far from what the hardware can do.)

**cuTile is close, but not first, on this GPU.** 135 TFLOP/s is 81% of peak from a 9-line kernel in which we did not even choose the number of threads. It is behind Triton and cuBLAS here, and that should not surprise us: cuTile has run on Ada only since March 2026, while Triton has been tuned for Ampere and Ada for years. We also used cuTile in its simplest form. NVIDIA's own matmul sample adds a block swizzle (a smarter order for visiting the output tiles, so that neighbouring blocks share data in the L2 cache) and per-GPU tuning.

We should also be careful about what this is not. It is one matrix size, one data type, one GPU, and tile sizes that we picked by hand. Change any of those, and the order may change.

---

## 8. When to Pick Which

**CUDA C++ (SIMT)**, what we used for most of this series, when:

* You need full control: exactly which thread does what, warp shuffles, atomics, custom synchronization.
* The problem is not tile-shaped: graphs, trees, sorting, irregular data, anything with lots of branching.
* You are writing a library or a kernel that must be the fastest possible on one GPU, and you are willing to pay for it with time (post 18).
* Your code is in a C++ application, and you do not want a Python runtime.

**Triton**, when:

* You work in Python and PyTorch, and the kernel is tile-shaped: matrix multiplies, attention, fused elementwise and reduction kernels.
* You want one kernel that runs on NVIDIA (8.0 and newer) and AMD GPUs.
* You want a large community and many examples. `torch.compile` writes Triton, so reading Triton is also a good way to understand what PyTorch does for you.

**cuTile**, when:

* You want the tile model from NVIDIA itself, with the compiler making even more decisions (no warps, no pointers).
* You target Hopper and Blackwell. There, the compiler can use TMA, tensor memory and the other hardware from post 20 without any change to your code. That is the main promise of CUDA Tile.
* You want tiles in C++ as well: since CUDA 13.3 the same model is available in CUDA C++, and tile kernels and ordinary SIMT kernels can live in the same file [5][16]. We have not run it here, because our toolkit is 13.1.

These are not exclusive. A real project often has most kernels from libraries (cuBLAS, CUB, cuDNN), a few tile kernels for the fused operations, and one or two hand-written CUDA C++ kernels for the odd cases. They even meet in the middle: there is an experimental Triton backend that compiles Triton to CUDA Tile IR instead of PTX [11]. It is currently validated only on Blackwell, so we could not try it.

And whatever you choose, everything from the earlier posts still applies. Coalescing, shared memory, occupancy, and the roofline are how you understand *why* a Triton or cuTile kernel is fast or slow. The compiler writes the threads for you; it does not stop the hardware from working the way it does.

---

## 9. Exercise

In both Python versions, we picked `BM, BN, BK = 128, 128, 32` and never changed it. Try a few other combinations, for example `(128, 128, 64)`, `(64, 128, 64)` and `(128, 256, 32)`, and time each one. In Triton, also try `num_warps=8` and `num_stages=2` or `4`.

You will find that some choices are much worse; in our quick try, one of them made cuTile more than twice as slow. Can you explain why using what you know from post 08 about registers and occupancy? This is why Triton has `@triton.autotune`, and why NVIDIA's cuTile samples set per-GPU hints such as `ct.ByTarget`: a tile language removes the thread-level work, but choosing the tile sizes is still yours.

---

## 10. Where to Go From Here

This is the last post of the series. We started with a kernel that printed "Hello from block 1, thread 0". Since then, we moved data to and from the GPU, measured and profiled our kernels, learned why coalescing and shared memory matter, wrote a reduction, a scan and a histogram, overlapped copies with streams, fed the tensor cores, built a matrix multiply step by step, fused a softmax, and looked at what Hopper and Blackwell add. And in this post, we saw that the same ideas now also come in a much shorter form.

If you want to go further, these are the resources we recommend:

* **Programming Massively Parallel Processors**, 4th edition, by Hwu, Kirk and El Hajj (2022) [12]. The memory and parallel patterns parts of this series follow it closely, and it goes much deeper.
* **GPU MODE** [13], a community that started as a reading group for that book. Its recorded lectures cover everything from the first chapters to FlashAttention, Triton and CUTLASS, and its Discord is where many kernel writers talk.
* **NVIDIA's CUDA C++ Programming Guide** [14] and **CUDA C++ Best Practices Guide** [15]. Now that you know the basics, they read very differently than on day one.

The best next step, though, is to take a slow piece of your own code, profile it, and make it faster.

In summary, tile programming lets us describe what a block does to a tile of data and leaves the threads to the compiler. On the RTX 4090, a vector add ran at the same speed in CUDA C++, Triton and cuTile, and a 14-line Triton matmul matched cuBLAS at 162 TFLOP/s, with a 9-line cuTile kernel at 135. In September 2026, cuTile runs on Ampere, Ada, Hopper and Blackwell, from Python and (since CUDA 13.3) C++, though the `tileiras` in CUDA 13.1 still only knows Blackwell. CUDA C++ remains the tool for full control and irregular problems, and the understanding you built with it is what makes you effective in any of the three.

[1]: https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/ "NVIDIA CUDA 13.1 Powers Next-Gen GPU Programming with NVIDIA CUDA Tile"
[2]: https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/ "CUDA 13.2 Introduces Enhanced CUDA Tile Support and New Python Features"
[3]: https://github.com/NVIDIA/cuda-tile/releases "NVIDIA/cuda-tile releases"
[4]: https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates/ "NVIDIA CUDA 13.3 Enhances GPU Development with Tile Programming in C++"
[5]: https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/ "Develop High-Performance GPU Kernels in C++ with NVIDIA CUDA Tile"
[6]: https://docs.nvidia.com/cuda/cutile-python/ "cuTile Python documentation"
[7]: https://github.com/NVIDIA/cutile-python "NVIDIA/cutile-python"
[8]: https://docs.nvidia.com/cuda/cutile-python/quickstart.html "cuTile Python Quickstart"
[9]: https://github.com/triton-lang/triton "Triton"
[10]: https://docs.nvidia.com/cuda/cutile-python/data.html "cuTile Python: Data Model"
[11]: https://github.com/triton-lang/Triton-to-tile-IR "Triton CUDA Tile IR backend"
[12]: https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0 "Programming Massively Parallel Processors, 4th edition"
[13]: https://github.com/gpu-mode/lectures "GPU MODE lectures"
[14]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "CUDA C++ Programming Guide"
[15]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ "CUDA C++ Best Practices Guide"
[16]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/tile-compilation-in-cuda.html "Tile Compilation in CUDA (nvcc)"
