---
title: "Post 11: CCCL and libraries: don't write what exists"
date: 2026-09-29T19:11:00+03:00
weight: 12
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the last two posts, we wrote a reduction, a scan, and a histogram by hand. That was worth doing once: it showed us how shared memory, warps, and atomics fit together. But at the end of post 09, one call to `cub::DeviceReduce` did the whole job. The same is true for most common building blocks. Sorting, reducing, scanning, matrix products, and FFTs already exist, they are tested, and they are tuned for every GPU generation by people at NVIDIA.

In this post, we tour the libraries that ship with the CUDA toolkit. Each demo is a few lines long. The goal is not to learn every function, but to know what exists, so that next time we reach for a library before we reach for `__global__`.

## What We Will Learn

In this article, we will cover:

* What CCCL is: Thrust, CUB, and libcu++ in one package.
* How to use Thrust like the STL: `device_vector`, `transform`, `reduce`, `sort`, and `transform_reduce` with a lambda.
* How CUB's two-call temporary storage pattern works, using `DeviceRadixSort`.
* How to use `cuda::atomic_ref` and `cuda::std::span` from libcu++ inside a kernel.
* How to call cuBLAS (SAXPY) and cuFFT (find the frequency of a sine).
* How a GPU sort compares with `std::sort`, with and without the copies.

## Table of Contents

1. The Libraries in the Toolkit.
2. The Full Program.
3. Thrust: The STL for the GPU.
4. cuBLAS: SAXPY in One Call.
5. Sorting: std::sort vs Thrust vs CUB.
6. libcu++: The Standard Library Inside Kernels.
7. cuFFT: Finding a Frequency.
8. Build and Run.
9. Exercise.

---

## 1. The Libraries in the Toolkit

CCCL (CUDA Core Compute Libraries) is three libraries that used to live in separate repositories and now ship together [1]:

| Library  | What it is                                      | Namespace              | Level          |
| -------- | ----------------------------------------------- | ---------------------- | -------------- |
| Thrust   | STL-like containers and algorithms              | `thrust::`             | Whole arrays   |
| CUB      | Fast building blocks for kernels and devices    | `cub::`                | Device, block, warp |
| libcu++  | The C++ standard library for host and device    | `cuda::std::`, `cuda::` | Single values  |

Thrust is the easiest to use and is built on top of CUB. CUB gives more control: we manage the temporary memory and the stream ourselves. libcu++ is what we use *inside* our own kernels.

Besides CCCL, the toolkit ships math libraries such as cuBLAS (linear algebra), cuFFT (Fourier transforms), cuRAND (random numbers), and cuSPARSE (sparse matrices). They are separate `.so` files that we link against.

CUDA 13.1 comes with CCCL 3.1. You can check the exact version with the `CCCL_VERSION` macro, and our program prints it:

```text
CCCL 3.1.4
```

CCCL 3.0 was a major release that removed many old names. If you follow an older tutorial, you may hit some of these [2]:

| Removed in CCCL 3.0             | Use instead               |
| ------------------------------- | ------------------------- |
| `cub::Sum`                      | `cuda::std::plus`         |
| `cub::Max`, `cub::Min`          | `cuda::maximum`, `cuda::minimum` |
| `thrust::identity<T>`           | `cuda::std::identity`     |
| `thrust::unary_function` base   | Nothing, just remove it   |
| `thrust::async::*`              | `thrust::cuda::par_nosync` policy |
| `debug_synchronous` argument of `cub::Device*` | Nothing, remove it |

Also, `thrust::plus` and the other Thrust function objects are now just aliases for the ones in `cuda::std::`, so we will use the `cuda::std::` names directly.

One more CUDA 13 change: the CCCL headers moved from `include/` to `include/cccl/` inside the toolkit. `nvcc` adds that directory for us, so `#include <thrust/sort.h>` keeps working in `.cu` files. Only files compiled by `g++` alone need it added, with `-I/usr/local/cuda/include/cccl` or by linking `CCCL::CCCL` in CMake. The migration guide warns not to write `#include <cccl/thrust/...>`.

---

## 2. The Full Program

```bash
mkdir cccl-libs
cd cccl-libs
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post11/main.cu" >}}

The program has four small demos, each in its own function that returns `0` on success. The includes tell us what we use:

{{< code file="cuda/post11/main.cu" region="includes" link="false" >}}

Thrust has one header per algorithm group. CUB has one header per device-wide algorithm. libcu++ headers follow the standard ones with a `cuda/std/` prefix, and the extensions that have no standard version, such as `cuda::atomic_ref` with a scope, live in `cuda/`.

At the top of the file, there is the same `-U_GNU_SOURCE` shim we added in post 09 for CUB. Thrust also includes `<shared_mutex>`, so it needs two more declarations:

{{< code file="cuda/post11/main.cu" region="shim" link="false" >}}

---

## 3. Thrust: The STL for the GPU

{{< code file="cuda/post11/main.cu" region="thrust" link="false" >}}

`thrust::device_vector` is a `std::vector` whose memory lives on the GPU. Its constructor calls `cudaMalloc` and fills the elements, and its destructor calls `cudaFree`. This is the same RAII idea as the `DeviceBuffer` we built with `unique_ptr` in [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}), so in this post we do not need `DeviceBuffer` at all. Constructing or assigning a `device_vector` from a `std::vector` copies the data to the GPU, and `thrust::copy` into a host iterator copies it back.

The algorithms look like the ones in `<algorithm>` and `<numeric>`. Because the iterators come from a `device_vector`, Thrust knows the data is on the GPU and runs a kernel. We did not write a `__global__` function, pick a block size, or compute a grid.

* `thrust::transform` applies a function to each element. With two input ranges it works like `std::transform` with a binary function. Here it computes SAXPY, `y = a * x + y`, a classic BLAS routine.
* `thrust::reduce` sums the elements, like `std::reduce`. The default is `+` starting from zero.
* `thrust::transform_reduce` does both in one pass: square each element, then add. The result is the squared length of `y`, and it never stores the squares in memory.

Reading `y[0]` also works, but be careful: each access to one element of a `device_vector` from the host is a separate `cudaMemcpy`. It is fine for one value, and very slow in a loop.

### Lambdas on the device

To pass a lambda to a Thrust algorithm, the lambda must be compiled for the GPU. We mark it with `__host__ __device__` and compile with `--extended-lambda`, which allows lambdas marked this way inside host functions.

Why `__host__ __device__` and not just `__device__`? The lambda only ever runs on the device. But some algorithms, such as `transform_reduce`, need to know the lambda's return type on the host, and `nvcc` cannot tell the host compiler what a `__device__`-only lambda returns. If we write `[] __device__ (float v)`, CCCL stops with a clear message:

```text
error: static assertion failed: Attempt to use an extended __device__ lambda in a context that requires querying its return type in host code. Use a named function object, an extended __host__ __device__ lambda, cuda::proclaim_return_type, or an extended __device__ lambda with a trailing return type inst...
```

The easiest rule is to always write `__host__ __device__` on lambdas that we pass to Thrust.

Output:

```text
thrust: y[0] = 4.0, sum = 4000000.0, norm = 4000.0
```

With `x = 1` and `y = 2`, every element of `y` becomes `2 * 1 + 2 = 4`. One million fours add up to 4,000,000, and the norm is `sqrt(1,000,000 * 16) = 4000`.

Thrust reports errors by throwing `thrust::system_error`, not by returning a `cudaError_t`. We do not catch it here, so an error ends the program with a message, which is fine for a demo.

---

## 4. cuBLAS: SAXPY in One Call

cuBLAS is NVIDIA's BLAS (Basic Linear Algebra Subprograms). We used it in post 07 for matrix multiply. SAXPY is its simplest routine, so let's compute the same thing again and check that both answers agree:

{{< code file="cuda/post11/main.cu" region="cublas" link="false" >}}

A few things to note:

* Every cuBLAS call needs a *handle*, created once with `cublasCreate`. It holds the library's state, such as the stream it runs on. In a real program, we create it once and reuse it, because creating it is not free.
* cuBLAS takes raw device pointers. `thrust::raw_pointer_cast` gets the `float*` out of a `device_vector`. This is how Thrust works together with every other CUDA library, and with our own kernels.
* `alpha` is passed by pointer. By default cuBLAS reads it from host memory.
* The `1`s are the *strides*: use every element. A stride of 2 would use every other one.
* cuBLAS returns a `cublasStatus_t`, not a `cudaError_t`, and `cublasGetStatusString` turns it into text. We destroy the handle before checking, so nothing leaks on failure.

Unlike post 07, there is no row-major trick here. SAXPY works on vectors, and a vector looks the same in both layouts.

`thrust::equal` compares the two results on the GPU:

```text
cublas: saxpy matches thrust: yes
```

---

## 5. Sorting: std::sort vs Thrust vs CUB

Sorting is where the GPU really shines, and where writing our own version would take the longest. We sort 16.7 million random 32-bit keys (64 MiB) three ways. First the CPU baseline:

{{< code file="cuda/post11/main.cu" region="std-sort" link="false" >}}

Then Thrust. We use the `time_ms` helper from post 03 (3 warm-up runs, average of 20) for the sort alone, and a CPU clock for the full trip: copy in, sort, and copy out.

{{< code file="cuda/post11/main.cu" region="thrust-sort" link="false" >}}

`thrust::sort` sees 32-bit unsigned keys and the default `<` order, so it picks a *radix sort*. A radix sort does not compare keys with each other. It sorts by a few bits at a time, starting from the lowest ones, and reads and writes the whole array in every pass. So it does the same work whether the input is random or already sorted. That is why the timing loop can keep sorting the same vector.

### CUB and temporary storage

Thrust's radix sort is really CUB's `DeviceRadixSort`. Let's call it directly:

{{< code file="cuda/post11/main.cu" region="cub-sort" link="false" >}}

This is the CUB pattern we first met with `cub::DeviceReduce` in post 09. CUB never allocates memory on its own. Instead, every device-wide algorithm is called twice:

1. With a null temporary pointer. CUB does nothing except write the number of bytes it needs into `temp_bytes`.
2. With a real buffer of that size. This call does the work.

CUB leaves the allocation to us because `cudaMalloc` and `cudaFree` are slow, and `cudaFree` waits for the whole GPU to finish. When we sort many times, we allocate once and reuse the buffer. Here the buffer is a `device_vector<char>` again, so we do not need to free it by hand.

CUB functions return `cudaError_t`, so `CUDA_CHECK` works on them. Inside the timing lambda we do not check each call, so we call `cudaGetLastError` afterwards, as in post 03.

Output (RTX 4090, Release build):

```text
sort 16777216 keys: all three match
  std::sort (1 CPU core):     951.9 ms
  thrust incl. copies:         13.8 ms
  thrust::sort:                1.75 ms
  cub::DeviceRadixSort:        0.64 ms (67621 KiB temp)
```

| Method                       | Time      | vs `std::sort` |
| ---------------------------- | --------- | -------------- |
| `std::sort`, 1 CPU core      | 951.9 ms  | 1x             |
| Thrust, with both copies     | 13.8 ms   | 69x            |
| `thrust::sort`, data on GPU  | 1.75 ms   | 544x           |
| `cub::DeviceRadixSort`       | 0.64 ms   | 1487x          |

There are three lessons in this table.

**The copies dominate.** If the data starts and ends on the CPU, about 12 of the 13.8 ms go to allocating GPU memory and moving 64 MiB in each direction over PCIe. The sort itself is the small part. The GPU still wins by 69 times, but the real win comes when the data *stays* on the GPU between steps. Post 12 shows how pinned memory and streams speed up and hide the copies.

**CUB is 2.7 times faster than Thrust here, for the same kernels.** The difference is the temporary memory. Each call to `thrust::sort` allocates its temporary buffer with `cudaMalloc` and releases it with `cudaFree`. An `nsys` trace (post 04) of our program shows a `cudaMalloc` and a `cudaFree` inside every timed `thrust::sort` call, and `cudaFree` waits for the GPU to go idle. In our CUB version, that buffer is allocated once, outside the timing loop. For a single sort this does not matter. For thousands of sorts in a loop, it does. Use CUB directly, or give Thrust your own allocator through its execution policy (`thrust::cuda::par(alloc)`).

**How fast is 0.64 ms?** Reading and writing 64 MiB once at the 4090's peak of 1008 GB/s takes `2 * 67.1 MB / 1008 GB/s = 0.133 ms`. The CUB sort takes 4.8 times that, so it costs about five trips through memory. That fits a radix sort that reads and writes the whole array once per group of key bits. We could not write this ourselves in an afternoon.

Where does the 66 MiB of temporary storage come from? We gave CUB separate input and output arrays and asked it not to touch the input, so it needs a third array to bounce the keys between passes. CUB also has an overload that takes a `cub::DoubleBuffer`, which lets it use both of our arrays for that. With it, the same sort asks for only 2085 KiB. The catch is that the result may end up in either array, and `DoubleBuffer::Current()` tells us which one.

---

## 6. libcu++: The Standard Library Inside Kernels

Sometimes no library call fits, and we write our own kernel. libcu++ gives us standard C++ types that work on the device, so our kernels can look like normal C++:

{{< code file="cuda/post11/main.cu" region="kernel" link="false" >}}

* `cuda::std::span<const float>` is `std::span` for the GPU: a pointer and a size in one small object that we pass by value. The kernel no longer needs a separate `n` argument, and the bounds check reads `i < v.size()`.
* `cuda::atomic_ref` wraps a plain `int` in device memory and gives it atomic operations, like C++20's `std::atomic_ref`. `fetch_add` does the same job as `atomicAdd`, but in the standard style, with an explicit memory order.

The second template argument is the *scope*: which threads must see the update. `thread_scope_block` only syncs with threads of the same block, `thread_scope_device` with all threads on this GPU, and `thread_scope_system` also with the CPU and other GPUs. A smaller scope can be cheaper. Here all blocks add to the same counter, so we need device scope. `cuda::std::atomic_ref` also exists, but, like the standard one, it always uses system scope.

libcu++ also has `cuda::atomic<T, scope>`, an atomic that owns its value, like `std::atomic`, and `cuda::std::array`, a fixed-size array that works in kernels.

Here is how we call it:

{{< code file="cuda/post11/main.cu" region="libcudacxx" link="false" >}}

`thrust::sequence` fills the vector with 0, 1, 2, and so on. We build the `span` from the raw pointer and the size, and pass it to the kernel. Then `thrust::count_if` counts the same thing, so we can check our kernel.

```text
libcu++: 250000 values above 749999.5 (count_if says 250000)
```

In real code, 250,000 threads all hitting one counter is slow, because the atomics on one address are done one at a time. The privatization trick from post 10 (count per block first, then add once per block) applies here too.

---

## 7. cuFFT: Finding a Frequency

A Fast Fourier Transform (FFT) turns a signal into the list of frequencies it contains. We build a sine wave that repeats 50 times over 1024 samples, transform it, and check that the strongest frequency is 50:

{{< code file="cuda/post11/main.cu" region="cufft" link="false" >}}

cuFFT works with *plans*. `cufftPlan1d` describes the transform (size 1024, complex to complex, one signal) and prepares everything it needs. `cufftExecC2C` runs it. Here it works in place: the output overwrites the input. Like cuBLAS handles, plans are expensive to create, so real code makes one and reuses it for every signal of the same size. `cufftHandle` is a plain integer, not a pointer, so we release it with `cufftDestroy` right after using it instead of wrapping it in a `unique_ptr`.

`cufftComplex` is a `float2` with `.x` as the real part and `.y` as the imaginary part. We use Thrust again to compute the magnitude of each bin with `cuCabsf`, then `thrust::max_element` to find the peak.

```text
cufft: peak at bin 50, magnitude 512.0 (expected bin 50, 512)
```

The peak is at bin 50, as expected. Why 512 and not 1024? A real sine is the sum of two complex waves, one at frequency +50 and one at -50. The FFT puts half of the signal in bin 50 and half in bin `1024 - 50 = 974`, its mirror. That is also why we only searched the first half. And cuFFT does not scale its output, so each half has a magnitude of `1024 / 2 = 512`. If you transform forward and then backward, you get the input multiplied by `n`, and it is up to you to divide.

---

## 8. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 --extended-lambda main.cu -o cccl-libs -lcublas -lcufft
./cccl-libs
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post11/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cccl-libs
```
{{< /tab >}}
{{< /tabs >}}

Thrust, CUB, and libcu++ are header-only, so there is nothing to link for them. cuBLAS and cuFFT are real libraries: `-lcublas -lcufft` with `nvcc`, and the `CUDA::cublas` and `CUDA::cufft` targets from `find_package(CUDAToolkit)` with CMake. The `--extended-lambda` flag is needed for our lambdas, and in CMake we add it only for CUDA files with a generator expression.

Notice `-DCMAKE_BUILD_TYPE=Release`. Without it, CMake compiles the host code without optimization. The GPU code does not care, but `std::sort` does: our CMake build without a build type took 2451.9 ms instead of about 950 ms. When you compare against the CPU, always compare against optimized CPU code.

Output:

```text
CCCL 3.1.4
thrust: y[0] = 4.0, sum = 4000000.0, norm = 4000.0
cublas: saxpy matches thrust: yes
sort 16777216 keys: all three match
  std::sort (1 CPU core):     951.9 ms
  thrust incl. copies:         13.8 ms
  thrust::sort:                1.75 ms
  cub::DeviceRadixSort:        0.64 ms (67621 KiB temp)
libcu++: 250000 values above 749999.5 (count_if says 250000)
cufft: peak at bin 50, magnitude 512.0 (expected bin 50, 512)
PASSED
```

---

## 9. Exercise

For 16.7 million keys, the GPU wins by 69 times even with the copies. But for a small array, the copies and the launch overhead cost more than `std::sort` itself.

Change `n` in `sort_demo` and find the *crossover*: the smallest size where "thrust incl. copies" beats `std::sort` on your machine. Try powers of two from `1 << 10` to `1 << 24`.

Then, as a second step, replace the CUB call with the `cub::DoubleBuffer` overload of `SortKeys`. Check that `temp_bytes` drops to about 2 MiB, compare the time, and use `d_keys.Current()` to find the sorted result.

Hint: every copy, allocation, and kernel launch has a fixed cost of some microseconds, no matter how small `n` is. `std::sort` on a tiny array has no such cost.

In summary, before writing a kernel, check whether it already exists. CCCL gives us three levels: Thrust for STL-style algorithms on whole arrays, CUB for the same algorithms with full control over memory and streams, and libcu++ for standard C++ types such as `span` and `atomic_ref` inside our own kernels. cuBLAS and cuFFT cover linear algebra and Fourier transforms, and `thrust::raw_pointer_cast` connects them all. A library sort beat `std::sort` by 1487 times on the GPU, and by 69 times including the copies, which tells us the next problem to solve. In the next post, we will make those copies faster with pinned memory and hide them behind computation with streams.

[1]: https://github.com/NVIDIA/cccl "CUDA Core Compute Libraries (CCCL)"
[2]: https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html "CCCL 2.x to CCCL 3.0 migration guide"
[3]: https://nvidia.github.io/cccl/unstable/thrust/ "Thrust documentation"
[4]: https://nvidia.github.io/cccl/unstable/cub/ "CUB documentation"
[5]: https://nvidia.github.io/cccl/unstable/libcudacxx/ "libcu++ documentation"
[6]: https://docs.nvidia.com/cuda/cublas/ "cuBLAS"
[7]: https://docs.nvidia.com/cuda/cufft/ "cuFFT"
