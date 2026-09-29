---
title: "Post 01: Moving data between the CPU and the GPU"
date: 2026-09-29T18:53:19+03:00
weight: 2
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the previous post, we wrote our first kernel and launched it on the GPU. It printed a few lines, but it did not compute anything. In this post, we will do some real work: add two vectors of one million numbers each on the GPU and bring the result back.

To do that, we first need to understand one important fact: the CPU and the GPU do not share memory. The GPU has its own memory (VRAM), and a kernel can only read and write data that lives there. So every CUDA program follows the same three steps:

1. Copy the input from the CPU (the *host*) to the GPU (the *device*).
2. Run the kernel.
3. Copy the result back from the device to the host.

## What We Will Learn

In this article, we will cover:

* How to allocate and free memory on the GPU.
* How to copy data between the host and the device.
* How each thread finds its own element.
* How to catch out-of-bounds bugs with `compute-sanitizer`.

## Table of Contents

1. The Full Program.
2. Allocating Device Memory.
3. Copying Data.
4. The Kernel.
5. Launching Enough Threads.
6. Build and Run.
7. Exercise.

---

## 1. The Full Program

```bash
mkdir vector-add
cd vector-add
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post01/main.cu" >}}

We keep the `CUDA_CHECK` macro from the previous post. Let's go through the new parts.

---

## 2. Allocating Device Memory

We start by preparing the input on the host as usual:

{{< code file="cuda/post01/main.cu" region="host" link="false" >}}

We use `std::vector` on the host. On the device, the equivalent of `malloc` and `free` are `cudaMalloc` and `cudaFree`:

```cpp
float* d_a = nullptr;
cudaMalloc(&d_a, bytes);
// use d_a
cudaFree(d_a);
```

The `d_` prefix is a common convention for pointers to device memory. It matters because both are plain `float*` to the compiler. If you dereference a device pointer on the host, the compiler will not stop you, and the program will crash.

There is a problem with the snippet above. Our `CUDA_CHECK` macro returns from `main` as soon as something fails. If that happens between `cudaMalloc` and `cudaFree`, the memory is never released. We have three buffers and several calls that can fail, so tracking this by hand quickly gets messy.

We have already solved this problem before. In [New way to use unique_ptr]({{< ref "/posts/new-way-to-use-unique_ptr" >}}), we used `std::unique_ptr` with `fclose` as a custom deleter to close a file automatically. `cudaFree` works exactly the same way:

{{< code file="cuda/post01/main.cu" region="buffer" link="false" >}}

`make_device_buffer` allocates the memory and hands it to a `unique_ptr` that will call `cudaFree` when it goes out of scope. If the allocation fails, it returns an empty `unique_ptr`, so we can check it with `operator bool`:

{{< code file="cuda/post01/main.cu" region="alloc" link="false" >}}

Now, no matter where `main` returns, the device memory is released. We use `.get()` whenever a CUDA function needs the raw pointer.

---

## 3. Copying Data

{{< code file="cuda/post01/main.cu" region="copy-in" link="false" >}}

`cudaMemcpy` works like `memcpy`, with one extra argument that tells it the direction. The order of the other arguments is the same as `memcpy`: destination first, then source, then the size in bytes.

| Direction                  | Meaning       |
| -------------------------- | ------------- |
| `cudaMemcpyHostToDevice`   | CPU to GPU    |
| `cudaMemcpyDeviceToHost`   | GPU to CPU    |
| `cudaMemcpyDeviceToDevice` | GPU to GPU    |

---

## 4. The Kernel

{{< code file="cuda/post01/main.cu" region="kernel" link="false" >}}

In the previous post, each thread only printed its `blockIdx.x` and `threadIdx.x`. Now we combine them into a single global index:

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

`blockDim.x` is the number of threads per block. With 256 threads per block, block 0 covers elements 0 to 255, block 1 covers 256 to 511, and so on. Each thread adds exactly one pair of numbers.

Notice there is no loop. On the CPU, we would write `for (int i = 0; i < n; ++i)`. On the GPU, the loop is replaced by one million threads, each one handling a single `i`.

---

## 5. Launching Enough Threads

{{< code file="cuda/post01/main.cu" region="launch" link="false" >}}

We pick 256 threads per block. It is a common default and a multiple of 32, the number of threads the GPU schedules together as a *warp*. Then we need enough blocks to cover all `n` elements.

`n / threads` is not enough, because integer division rounds down. One million divided by 256 is 3906.25, so 3906 blocks would leave the last 64 elements unprocessed. The expression `(n + threads - 1) / threads` rounds up instead, giving us 3907 blocks.

This rounding gives us 3907 × 256 = 1,000,192 threads, which is 192 more than we have elements. That is why the kernel has the `if (i < n)` check: the extra threads simply do nothing.

Finally, we copy the result back:

{{< code file="cuda/post01/main.cu" region="copy-out" link="false" >}}

Unlike the previous post, we did not call `cudaDeviceSynchronize`. We do not need it here, because `cudaMemcpy` waits for the kernel to finish before it starts copying.

---

## 6. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE main.cu -o vector-add
./vector-add
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post01/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/vector-add
```
{{< /tab >}}
{{< /tabs >}}

If you are wondering about `-U_GNU_SOURCE`, it is the workaround for the glibc and CUDA 13.1 header clash we covered in the previous post.

Output:

```text
Launched 3907 blocks x 256 threads for 1000000 elements
PASSED (0 errors)
```

We check every element on the CPU:

{{< code file="cuda/post01/main.cu" region="verify" link="false" >}}

Since `a[i] = i` and `b[i] = 2 * i`, every `c[i]` must equal `3 * i`. All of these values are below 2^24, so a `float` holds them exactly and comparing with `!=` is safe here.

---

## 7. Exercise

Remove the `if (i < n)` check from the kernel, then build and run the program again.

Surprisingly, it still prints `PASSED`. The 192 extra threads read and write past the end of our buffers, but they happen not to hit anything important this time. This is the worst kind of bug: the program looks correct until one day it isn't.

The CUDA toolkit comes with a tool that catches exactly this. Run the program under `compute-sanitizer`:

```bash
compute-sanitizer ./vector-add
```

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at add(const float *, const float *, float *, int)+0x80
=========     by thread (96,0,0) in block (3906,0,0)
=========     Access to 0x79fd33fd0980 is out of bounds
=========     and is 129 bytes after the nearest allocation at 0x79fd33c00000 of size 4,000,000 bytes
```

It tells us the kernel, the block, the thread, and how far past the allocation we went. Thread 96 of block 3906 is global index 3906 × 256 + 96 = 1,000,032, well past the end of our one million elements. Put the check back and run it again. You should see `ERROR SUMMARY: 0 errors`.

In summary, a GPU program is mostly about moving data: allocate with `cudaMalloc`, copy with `cudaMemcpy`, and free with `cudaFree`, which `unique_ptr` can do for us. Each thread computes its own index from `blockIdx`, `blockDim`, and `threadIdx`, and we launch enough blocks to cover everything, with a bounds check for the extra threads. Make a habit of running new kernels under `compute-sanitizer`; a `PASSED` alone does not prove there is no bug.

[1]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html "CUDA Runtime API: Memory Management"
[2]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html "Compute Sanitizer"
