---
title: "Post 00: Helloworld CUDA in 2026"
date: 2026-09-29T18:41:58+03:00
weight: 1
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In this series, we will have a beginner-friendly introduction to GPU programming with CUDA.

CUDA is NVIDIA's platform for running general-purpose code on the GPU. You write a function in C++, mark it as a *kernel*, and the GPU runs thousands of copies of it in parallel. Most of the tutorials you find online were written for CUDA 10 or 11, and a few things have changed since then. CUDA 13 dropped support for older GPUs, Ubuntu now ships the toolkit in its own archive, and CMake can detect your GPU for you. So in this first post, we will write the classic hello world, but the 2026 way.

## What We Will Learn

In this article, we will cover:

* How to install the CUDA toolkit on Ubuntu 26.04.
* How to write and launch our first kernel.
* How to build it with `nvcc` and with CMake.
* How to work around a header clash between CUDA 13.1 and recent glibc.

## Table of Contents

1. Install CUDA.
2. Validate the Installation.
3. Write the Hello World Kernel.
4. Build and Run.
5. Exercise.

---

## 1. Install CUDA

Starting with Ubuntu 26.04, the CUDA toolkit is available directly from the `multiverse` repository, so we no longer need to add NVIDIA's apt repository.

```bash
sudo apt update
sudo apt install cuda-toolkit
```

Be careful with the name. There is also an older package called `nvidia-cuda-toolkit`, but it is still on CUDA 12.4. We want `cuda-toolkit`, which gives us CUDA 13.1.

| Package              | Version | Description                              |
| -------------------- | ------- | ---------------------------------------- |
| cuda-toolkit         | 13.1    | The one we want                          |
| nvidia-cuda-toolkit  | 12.4    | Older Debian packaging `You can skip it` |

The toolkit is installed under `/usr/local/cuda`, which is not on your `PATH` by default. Add it to your `~/.bashrc` (or `~/.zshrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

If you are on another distribution, follow the official installation guide [1].

---

## 2. Validate the Installation

We need two things to work: the driver, which talks to the GPU, and the compiler, `nvcc`, which builds our code.

```bash
nvidia-smi
nvcc --version
```

Example output: (Ubuntu 26.04, RTX 4090)

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0
```

One more thing to check is your GPU's *compute capability*, which is NVIDIA's version number for the GPU architecture:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

```text
name, compute_cap
NVIDIA GeForce RTX 4090, 8.9
```

CUDA 13 dropped Maxwell, Pascal, and Volta GPUs, so it needs a compute capability of at least 7.5 (Turing, the RTX 20 series). If your number is lower, you will need to stay on CUDA 12 [2].

---

## 3. Write the Hello World Kernel

```bash
mkdir hello-cuda
cd hello-cuda
code main.cu
```

CUDA source files use the `.cu` extension. You can write the following in `main.cu`

{{< code file="cuda/post00/main.cu" >}}

Let's go through it piece by piece.

### The kernel

{{< code file="cuda/post00/main.cu" region="kernel" link="false" >}}

The `__global__` keyword turns a normal function into a kernel. It is called from the CPU (the *host*) but runs on the GPU (the *device*). A kernel must return `void`.

Every copy of the kernel runs as a *thread*, and threads are grouped into *blocks*. Inside the kernel, `blockIdx.x` tells us which block we are in and `threadIdx.x` tells us which thread we are within that block. These two built-in variables are how each thread figures out which piece of work is its own.

### Launching the kernel

{{< code file="cuda/post00/main.cu" region="launch" link="false" >}}

The `<<<2, 4>>>` syntax is CUDA's extension to C++. It means *launch 2 blocks with 4 threads each*, so our kernel runs 8 times.

A kernel launch is asynchronous. The CPU queues the work and continues immediately without waiting for the GPU. That is why we need two calls after it:

* `cudaGetLastError` catches problems with the launch itself, such as asking for too many threads.
* `cudaDeviceSynchronize` waits until the GPU finishes, and returns any error that happened while the kernel was running. It also flushes the output of `printf` from the device. Without it, `main` may return before anything gets printed.

### Checking errors

{{< code file="cuda/post00/main.cu" region="check" link="false" >}}

Almost every CUDA runtime function returns a `cudaError_t`, and CUDA does not throw exceptions. If you ignore the return value, a failure goes unnoticed and your program keeps running on wrong data. The `CUDA_CHECK` macro prints the file, the line, and a readable message, then exits `main`. You will see this macro, or something very similar, in almost every CUDA codebase.

### Querying the device

{{< code file="cuda/post00/main.cu" region="device" link="false" >}}

This is not strictly needed for a hello world, but it confirms which GPU we are actually running on. It is useful when a machine has more than one.

---

## 4. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE main.cu -o hello-cuda
./hello-cuda
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post00/CMakeLists.txt" >}}

Setting `CMAKE_CUDA_ARCHITECTURES` to `native` (CMake 3.24 and later) compiles for the GPU installed in your machine. Before that, you had to look up the compute capability and write it yourself.

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/hello-cuda
```
{{< /tab >}}
{{< /tabs >}}

Output:

```text
Running on NVIDIA GeForce RTX 4090 (compute capability 8.9)
Hello from block 1, thread 0
Hello from block 1, thread 1
Hello from block 1, thread 2
Hello from block 1, thread 3
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
Hello from block 0, thread 3
```

Notice that block 1 printed before block 0. The GPU does not promise any order between blocks, and your code should never depend on one. Run it a few times and the order may change.

### What is `-U_GNU_SOURCE`?

If you compile without that flag on Ubuntu 26.04, you will get this error:

```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrt" (declared at line 629 of /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h)
```

`rsqrt` (reciprocal square root) was added to C in the C23 standard, so recent versions of glibc now declare it in `math.h`. CUDA has had its own `rsqrt` for years, and CUDA 13.1 declares it slightly differently. When both declarations end up in the same file, the compiler refuses to continue.

glibc only exposes C23 extensions like this one when `_GNU_SOURCE` is defined, and `g++` defines it by default. Undefining it with `-U_GNU_SOURCE` hides glibc's declaration and leaves CUDA's version in place. Treat this as a temporary workaround: once a newer toolkit fixes the header, you can drop the flag.

---

## 5. Exercise

Change the launch to `hello<<<1, 2048>>>()` and run the program again. What happens, and which of the two checks catches it?

Hint: a block can have at most 1024 threads.

In summary, a CUDA hello world in 2026 is still a `__global__` function and a `<<<blocks, threads>>>` launch, but the tooling around it has changed: the toolkit comes from Ubuntu's archive, CMake picks the GPU architecture for you, and CUDA 13 needs a Turing GPU or newer. In the next post, we will stop printing and start computing, moving data between the CPU and GPU memory.

[1]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ "CUDA Installation Guide for Linux"
[2]: https://developer.nvidia.com/cuda-gpus "CUDA GPU Compute Capability"
