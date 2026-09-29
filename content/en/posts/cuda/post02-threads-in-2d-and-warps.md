---
title: "Post 02: Threads in 2D and warps"
date: 2026-09-29T19:02:00+03:00
weight: 3
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the [previous post]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}), we added two vectors on the GPU. Each thread computed one index, `blockIdx.x * blockDim.x + threadIdx.x`, and handled one element. That works well for a flat array, but a lot of real data is not flat. Images, matrices, and grids all have rows and columns.

In this post, we will blur a grayscale image on the GPU. We will give each thread an `(x, y)` position instead of a single index. Then we will look at how the GPU actually runs those threads: in groups of 32 called *warps*. Knowing about warps explains why an innocent-looking `if` can make a kernel twice as slow.

## What We Will Learn

In this article, we will cover:

* How to launch a 2D grid of 2D blocks with `dim3`.
* How each thread finds its pixel in an image.
* How a grid-stride loop lets a fixed number of threads handle an image of any size.
* What a warp is, and how threads in a 2D block are grouped into warps.
* What branch divergence is, and how much it costs.

## Table of Contents

1. The Full Program.
2. The Image.
3. 2D Blocks and Grids.
4. The Blur Kernel.
5. Grid-Stride Loops.
6. Warps.
7. Branch Divergence.
8. Build and Run.
9. Exercise.

---

## 1. The Full Program

```bash
mkdir blur
cd blur
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post02/main.cu" >}}

It is longer than the previous one, but most of it is the same pattern we already know: prepare data on the host, copy it to the device, launch a kernel, copy the result back, and compare it with the CPU. The `CUDA_CHECK` macro and `make_device_buffer` are unchanged from the previous post. Let's go through the new parts.

---

## 2. The Image

We do not want to deal with image files and libraries in this post, so we generate a grayscale image in code:

{{< code file="cuda/post02/main.cu" region="image" link="false" >}}

The image is 4000 × 3000 pixels, the size of a 12-megapixel photo. Each pixel is a brightness value between 0 and 255. We store it as `float`, so we can keep using the `DeviceBuffer` from the previous post. `(x ^ y) & 0xFF` gives a classic XOR pattern, full of sharp edges, which is a good test for a blur.

The image is stored *row by row* in one flat array. This is called *row-major* order. Pixel `(x, y)` lives at index `y * width + x`: skip `y` full rows, then move `x` pixels into the row. Keep this formula in mind, because every kernel in this post uses it.

---

## 3. 2D Blocks and Grids

So far, we launched kernels with two plain integers, `<<<blocks, threads>>>`. Both of these can also be a `dim3`, a small struct with three fields: `x`, `y`, and `z`. Any field you leave out is 1.

{{< code file="cuda/post02/main.cu" region="launch2d" link="false" >}}

`dim3 block(16, 16)` means each block is a 16 × 16 square of threads, 256 threads in total. That is the same thread count as the previous post, just arranged as a square.

The grid is computed the same way as before, but once per dimension. We need enough blocks across to cover 4000 columns, and enough blocks down to cover 3000 rows:

| Dimension | Pixels | Block size | Blocks                                 |
| --------- | ------ | ---------- | -------------------------------------- |
| x         | 4000   | 16         | 4000 / 16 = 250                        |
| y         | 3000   | 16         | 3000 / 16 = 187.5, rounded up to 188   |

So we launch a 250 × 188 grid, 47,000 blocks in total. The last row of blocks sticks out 8 rows below the image, which is why the kernel still needs a bounds check.

We read the block shape from the command line, so we can try different shapes in the exercise without recompiling. `./blur 32 8` runs with 32 × 8 blocks.

---

## 4. The Blur Kernel

A box blur replaces every pixel with the average of the pixels around it. We use a 5 × 5 box, so each output pixel is the average of 25 input pixels. Near the edges, some of these neighbours are outside the image, so we only average the ones that exist.

{{< code file="cuda/post02/main.cu" region="pixel" link="false" >}}

This function is marked `__host__ __device__`. That tells `nvcc` to compile it twice: once for the CPU and once for the GPU. A `__device__` function can be called from a kernel, but not launched on its own. Adding `__host__` lets us call the exact same code on the CPU to build our reference result:

{{< code file="cuda/post02/main.cu" region="reference" link="false" >}}

Since the CPU and the GPU add the same numbers in the same order, the results must match exactly, so we can compare them with `!=` like in the previous post.

Now the kernel itself:

{{< code file="cuda/post02/main.cu" region="blur2d" link="false" >}}

This is the previous post's index formula, written once for `x` and once for `y`. `blockIdx`, `blockDim`, and `threadIdx` all have `.x`, `.y`, and `.z` fields. Thread `(3, 5)` of block `(2, 1)` with 16 × 16 blocks works on pixel `x = 2 * 16 + 3 = 35`, `y = 1 * 16 + 5 = 21`.

The bounds check now covers both directions. The threads in the extra 8 rows at the bottom of the grid do nothing.

---

## 5. Grid-Stride Loops

In `blur2d`, we launch one thread per pixel. That means the grid size depends on the image size. There is another common style, where the grid size is fixed and each thread loops over as many pixels as needed:

{{< code file="cuda/post02/main.cu" region="stride" link="false" >}}

Each thread starts at the same position as in `blur2d`. After it finishes that pixel, it jumps ahead by the width of the whole grid, `blockDim.x * gridDim.x`, and handles the next one. `gridDim` is the number of blocks in the grid, the same value we passed as the first launch argument. The same goes for rows. It is called a *grid-stride loop*, because the stride is the size of the whole grid.

Note that there is no separate bounds check anymore. The loop conditions `y < height` and `x < width` already stop the threads that fall outside the image.

To prove that this works for any grid size, we launch only 8 × 8 blocks:

{{< code file="cuda/post02/main.cu" region="launchstride" link="false" >}}

That is 64 blocks of 256 threads, 16,384 threads for 12 million pixels, so each thread blurs about 730 pixels. We clear the output with `cudaMemset` first, so we know that the `PASSED` we see comes from this kernel and not from what `blur2d` left behind.

Why would we want this? A grid-stride kernel works for any input size with any grid, so you never have to worry about a grid that is too large. It also makes it easy to choose the grid based on the GPU instead of the data. Many library kernels are written this way.

But correct is not the same as fast. On our RTX 4090, this 8 × 8 grid takes about 1.0 ms, while `blur2d` takes 0.11 ms. The 4090 has 128 *streaming multiprocessors* (SMs), the units that actually run blocks. With only 64 blocks, at least half of them have nothing to do. When we changed the grid to 32 × 32 (1024 blocks), the time dropped to 0.14 ms. A grid-stride loop is a safety net, not an excuse to launch too few threads.

---

## 6. Warps

So far, we have talked about threads as if each one runs on its own. That is not how the hardware works. An SM runs threads in groups of 32 called *warps*. All 32 threads in a warp execute the same instruction at the same time, each on its own data. One instruction, 32 results.

The GPU splits a block into warps by counting threads in order: `x` first, then `y`, then `z`. The position of a thread in its block is:

```text
linear = threadIdx.y * blockDim.x + threadIdx.x
warp   = linear / 32
```

For our 16 × 16 blocks, this gives:

| Warp | Threads (`threadIdx.y`) | Pixels it covers                  |
| ---- | ----------------------- | --------------------------------- |
| 0    | rows 0 and 1            | 2 rows × 16 pixels                |
| 1    | rows 2 and 3            | 2 rows × 16 pixels                |
| ...  | ...                     | ...                               |
| 7    | rows 14 and 15          | 2 rows × 16 pixels                |

So a warp in a 16 × 16 block is a strip of 16 × 2 pixels. With 32 × 8 blocks, a warp is exactly one row of 32 pixels. With 8 × 8 blocks, it is 4 rows of 8 pixels. This will matter in the exercise.

This is also why block sizes are almost always a multiple of 32. A block of 100 threads still uses 4 warps (128 slots), and 28 of those slots do nothing.

---

## 7. Branch Divergence

If all 32 threads in a warp run the same instruction, what happens with an `if` where some threads go one way and some go the other?

The warp runs *both* paths, one after the other. While it runs the `if` part, the threads that took the `else` part sit idle, and then the other way around. This is called *branch divergence*. If the threads of a warp all take the same path, there is no problem: the warp just skips the other one.

We can measure it. This kernel does the same amount of work on every thread. Half of the threads take path A, and the other half take path B:

{{< code file="cuda/post02/main.cu" region="diverge" link="false" >}}

The only difference between the two runs is *which* threads take which path:

* `group = 32`: threads 0 to 31 take path A, threads 32 to 63 take path B, and so on. Every warp agrees.
* `group = 1`: even threads take path A and odd threads take path B. Every warp is split in half.

In both cases, exactly half the threads do A and half do B.

{{< code file="cuda/post02/main.cu" region="launchdiverge" link="false" >}}

To see the difference, we need to time the kernel. We use this small helper:

{{< code file="cuda/post02/main.cu" region="timer" link="false" >}}

It records a GPU timestamp before and after 100 launches and returns the average. We will explain `cudaEvent_t` and how to time kernels properly in post 03; for now, trust the numbers. We use the same helper for the two blur kernels.

On our RTX 4090, the warp-uniform version takes 2.32 ms and the divergent version takes 4.50 ms. Same work, same number of threads, almost twice the time, only because of how the threads are grouped.

Does this mean every `if` is bad? No. Divergence only costs time when threads *in the same warp* disagree, and only as much as the extra path costs. Our blur kernel has two `if`s: the bounds check and the edge check in `blur_pixel`. Both only disagree in the few warps that touch the image border, which is a tiny fraction of the 12 million pixels. The lesson is to avoid branches that split *every* warp, like `threadIdx.x % 2`.

---

## 8. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o blur
./blur
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post02/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/blur
```
{{< /tab >}}
{{< /tabs >}}

We add two flags to the `nvcc` command this time. `-arch=native` compiles the kernels for the GPU in your machine, the same thing `CMAKE_CUDA_ARCHITECTURES native` does in CMake. Now that we measure time, we want the code built for our exact GPU. `-O3` optimizes the host code, which makes the CPU reference blur faster; the kernels are optimized by default.

Output:

```text
Image 4000 x 3000, 5x5 box blur
blur2d:           grid  250 x 188  block 16 x 16  0.114 ms  PASSED
blur_grid_stride: grid    8 x 8    block 16 x 16  1.009 ms  PASSED
branchy group=32  2.323 ms
branchy group=1   4.498 ms
```

Your numbers will be different on a different GPU, but the pattern should be the same: the small grid-stride grid is much slower than one thread per pixel, and the divergent `branchy` is close to twice as slow as the uniform one.

As in the previous post, it is a good habit to run a new program under `compute-sanitizer`. It should end with `ERROR SUMMARY: 0 errors`.

---

## 9. Exercise

Run the blur with different block shapes and compare the times:

```bash
./blur 8 8
./blur 16 16
./blur 32 8
```

All three have a multiple of 32 threads per block and all of them print `PASSED`. Before you look at the results below, try to guess which one is fastest, using the warp table from section 6. Then try a few shapes of your own, such as `32 32`, `256 1`, and `1 256`.

Here is what we got on the RTX 4090:

```text
blur2d:           grid  500 x 375  block  8 x 8   0.129 ms  PASSED
blur2d:           grid  250 x 188  block 16 x 16  0.112 ms  PASSED
blur2d:           grid  125 x 375  block 32 x 8   0.108 ms  PASSED
blur2d:           grid  125 x 94   block 32 x 32  0.181 ms  PASSED
blur2d:           grid   16 x 3000 block 256 x 1   0.110 ms  PASSED
blur2d:           grid 4000 x 12   block  1 x 256  1.657 ms  PASSED
```

| Block    | Threads | One warp covers   | Time     |
| -------- | ------- | ----------------- | -------- |
| 8 × 8    | 64      | 4 rows × 8 pixels | 0.129 ms |
| 16 × 16  | 256     | 2 rows × 16 pixels| 0.112 ms |
| 32 × 8   | 256     | 1 row × 32 pixels | 0.108 ms |
| 32 × 32  | 1024    | 1 row × 32 pixels | 0.181 ms |
| 256 × 1  | 256     | 1 row × 32 pixels | 0.110 ms |
| 1 × 256  | 256     | 32 rows × 1 pixel | 1.657 ms |

The wider the strip a warp covers in `x`, the faster the blur. Pixels next to each other in a row are next to each other in memory. When a warp reads 32 neighbouring floats, the GPU can fetch them together in one go. When the same warp reads 4 short pieces from 4 different rows, it needs more separate memory accesses. `1 × 256` is the extreme case: each warp reads one pixel from 32 different rows, and it is 15 times slower. This idea is called *memory coalescing*, and it gets a full post of its own in post 05.

`32 × 32` is the odd one out. Its warps have the good shape, yet it is the second slowest. Blocks of 1024 threads are big, and an SM can only hold a limited number of threads at once. When a block does not fit neatly, part of the SM stays empty. We will come back to this when we talk about occupancy in post 08.

For now, the practical rule is: make `blockDim.x` at least 32, so each warp works on one contiguous piece of a row, and keep blocks at a moderate size like 256 threads.

In summary, `dim3` lets us launch 2D (or 3D) grids of 2D blocks, and each thread computes its `(x, y)` from `blockIdx`, `blockDim`, and `threadIdx`, with a bounds check in both directions. A grid-stride loop decouples the grid size from the data size, but the grid still needs enough blocks to keep every SM busy. Under the hood, the GPU runs threads in warps of 32, which run one instruction at a time for all 32 threads. Branches that split a warp make it run both paths, and the shape of a block decides which pixels each warp touches. In the next post, we will look at how to measure speed properly, and what the numbers we printed here really mean.

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy "CUDA C++ Programming Guide: Thread Hierarchy"
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture "CUDA C++ Programming Guide: SIMT Architecture"
[3]: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/ "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
