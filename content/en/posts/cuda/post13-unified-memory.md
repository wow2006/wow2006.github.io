---
title: "Post 13: Unified memory"
date: 2026-09-29T19:13:00+03:00
weight: 14
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

Since [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}), every program in this series has had two copies of its data: a `std::vector` on the host and a `cudaMalloc` buffer on the device, with `cudaMemcpy` in between. In post 12, we made those copies faster with pinned memory and streams. But we still had to write them by hand.

CUDA has another option. *Unified memory* gives us one pointer that works on both the CPU and the GPU. We allocate it once, and the CUDA driver moves the data to whichever side is using it. No `cudaMemcpy` at all. That sounds like a free lunch, and for correctness it is: the program gives the right answer. For speed, it depends a lot on *when* the data moves. In this post, we will measure that and see when unified memory is convenient and when it quietly makes a kernel 160 times slower.

## What We Will Learn

In this article, we will cover:

* How to allocate unified memory with `cudaMallocManaged`.
* How the data moves: page faults and on-demand migration on Linux.
* How to move the data ahead of time with `cudaMemPrefetchAsync` (and its new CUDA 13 signature).
* How to give the driver hints with `cudaMemAdvise`.
* How to see page faults and migrations in `nsys`.
* When unified memory helps, and when it silently hurts.

## Table of Contents

1. The Full Program.
2. Managed Memory.
3. What Happens on the First Kernel.
4. Prefetching.
5. Hints with `cudaMemAdvise`.
6. Build and Run.
7. Seeing the Page Faults in `nsys`.
8. Convenient or Costly?
9. Exercise.

---

## 1. The Full Program

```bash
mkdir unified-memory
cd unified-memory
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post13/main.cu" >}}

We use the same vector add as in post 01, on bigger arrays: 64 Mi floats, or 256 MB per array. That is much larger than the 72 MB L2 cache of the RTX 4090, so the kernel really reads from memory. The program runs the same kernel on four kinds of memory and prints how long the kernel took, and how long the work before it took (a copy or a prefetch). You can run a single case by passing its number, for example `./unified-memory 2`. We will use that with `nsys` later.

One thing is different from post 03. There, our `time_ms` helper ran the kernel a few times as a warm-up and then averaged 20 runs. We cannot do that here: the first run is the one that moves the data, so a warm-up would hide exactly what we want to see. Every case times a single launch, on freshly allocated memory.

---

## 2. Managed Memory

{{< code file="cuda/post13/main.cu" region="buffer" link="false" >}}

`cudaMallocManaged` has the same shape as `cudaMalloc`, and it is freed with the same `cudaFree`, so our `unique_ptr` deleter from post 01 works for both. The memory it returns is called *managed* memory. The difference is that the pointer is valid on the host too:

{{< code file="cuda/post13/main.cu" region="managed" link="false" >}}

The CPU fills `a` and `b` directly with a normal loop, the kernel reads them, and the CPU reads `c` to check the result. There is no `std::vector`, no `d_` prefix, and no `cudaMemcpy`. That is the whole appeal.

There is one rule we must not forget: a kernel launch is still asynchronous. The CPU must wait for the kernel (here with `cudaDeviceSynchronize`) before it reads `c`. Otherwise, it reads the result while the GPU is still writing it.

Notice also that we launch the same kernel twice and time both launches. We will see why in a moment.

---

## 3. What Happens on the First Kernel

Managed memory is not magic shared RAM. The data still lives in one place at a time: either in CPU memory or in GPU memory. The driver moves it in *pages*, the same unit the operating system uses for virtual memory.

On Linux, with a GPU from Pascal (2016) or newer, the process looks like this:

1. `cudaMallocManaged` only reserves addresses. No page exists yet.
2. The CPU writes `a[i] = i`. The pages are created in CPU memory, because that is where they were touched first.
3. The kernel starts and a warp reads `a[i]`. The page is not in GPU memory, so the GPU raises a *page fault*. The warp stops and waits.
4. The fault goes to the CUDA driver on the CPU (the `nvidia-uvm` kernel module). It collects a batch of faults, copies the pages over PCIe, updates the GPU page tables, and tells the GPU to retry.
5. The warp continues. The next warp that touches a page that is still on the CPU faults again.

So the first kernel does two jobs at once: the vector add, and moving 512 MB over PCIe in thousands of small pieces, with a round trip to the CPU for every batch. The time of that kernel includes all of it.

The program prints two device attributes that tell us which behavior we get:

{{< code file="cuda/post13/main.cu" region="query" link="false" >}}

`concurrentManagedAccess = 1` means the GPU supports page faults, so the CPU and the GPU can both touch managed memory, and data moves on demand. It is `1` on Linux with any GPU that CUDA 13 supports. On Windows it is `0`. There, the driver moves all managed memory to the GPU at every kernel launch, and the CPU must not touch it at all while any kernel is running.

`pageableMemoryAccess = 0` means the GPU cannot read normal `malloc` memory directly. On systems where it is `1` (for example Linux with Heterogeneous Memory Management, HMM, enabled in the driver, or Grace Hopper), even plain `new` and `malloc` memory behaves like managed memory. On our machine, only `cudaMallocManaged` gets this treatment.

---

## 4. Prefetching

If we know the kernel will need the data, we do not have to wait for page faults. We can tell the driver to move it first:

{{< code file="cuda/post13/main.cu" region="prefetch" link="false" >}}

`cudaMemPrefetchAsync` moves a range of managed memory to a location, in a stream, and does not block the CPU. The location is described by a `cudaMemLocation`:

{{< code file="cuda/post13/main.cu" region="locations" link="false" >}}

The prefetch moves the pages in large, 2 MB pieces, with no faults, so it uses PCIe much better. Then the kernel starts with everything already in GPU memory and runs at full speed. Prefetching `c`, which nobody has touched yet, does not copy anything: it only creates its pages directly in GPU memory.

### The CUDA 13 signature

If you copy `cudaMemPrefetchAsync` from an older tutorial, it will not compile with CUDA 13:

```cpp
cudaMemPrefetchAsync(ptr, bytes, 0);               // CUDA 12: destination device 0
cudaMemPrefetchAsync(ptr, bytes, gpu, 0, stream);  // CUDA 13: location, flags, stream
```

```text
error: no suitable constructor exists to convert from "int" to "cudaMemLocation"
```

In CUDA 12, the destination was a plain `int`: a device number, or `cudaCpuDeviceId` for the host. CUDA 12.2 added new versions that take a `cudaMemLocation` instead, and CUDA 13.0 made those the only ones. The `flags` argument must be `0` for now. `cudaMemAdvise` changed the same way: its last argument is now a `cudaMemLocation` instead of a device number.

---

## 5. Hints with `cudaMemAdvise`

A prefetch moves data once. `cudaMemAdvise` sets a longer-lasting *policy* for a range of managed memory. There are three kinds of advice:

| Advice                              | Meaning                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| `cudaMemAdviseSetReadMostly`        | Mostly read. A reader gets its own read-only copy, so both sides can read fast.  |
| `cudaMemAdviseSetPreferredLocation` | Keep the pages here when possible. Other processors map them instead of moving them. |
| `cudaMemAdviseSetAccessedBy`        | This processor will access the range, so keep it mapped there and avoid faults.  |

Each one has an `Unset` version to undo it. In the last case of our program, we combine two of them:

{{< code file="cuda/post13/main.cu" region="advise" link="false" >}}

This says: keep the data in CPU memory, and let the GPU access it where it is. Now the pages never move. When the kernel reads `a[i]`, the GPU reads it over PCIe, straight from CPU memory. This is the same idea as *zero-copy* memory, but with managed memory.

---

## 6. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o unified-memory
./unified-memory
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post13/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/unified-memory
```
{{< /tab >}}
{{< /tabs >}}

Output: (RTX 4090, PCIe 4.0)

```text
concurrentManagedAccess = 1, pageableMemoryAccess = 0
3 arrays x 256 MB, 262144 blocks x 256 threads

case                                before       kernel   bandwidth (% peak)
cudaMalloc + cudaMemcpy           53.25 ms     0.836 ms    964 GB/s ( 95.6%)  PASSED
managed, page faults                     -   135.298 ms      6 GB/s (  0.6%)  PASSED
managed, same data again                 -     0.859 ms    938 GB/s ( 93.0%)  PASSED
managed + cudaMemPrefetchAsync    26.26 ms     0.831 ms    969 GB/s ( 96.1%)  PASSED
managed, stays on CPU (advise)           -    50.866 ms     16 GB/s (  1.6%)  PASSED
```

Every case prints `PASSED`. Unified memory never gives a wrong answer; it only changes the time. Let's read the numbers:

* **`cudaMalloc` + `cudaMemcpy`** is our baseline from post 01. Copying 512 MB of input takes 53 ms, and the kernel runs at 96% of the 1008 GB/s peak we computed in post 03.
* **Page faults.** The same kernel on managed memory takes 135 ms, about 160 times slower. It even takes longer than the copy and the kernel of the baseline together (54 ms). The kernel code did not change at all; the time went into page faults. Nothing in the program tells you this happened, except the clock.
* **Same data again.** The second launch on the same arrays is fast again, because the pages are already in GPU memory. The fault cost is paid once per migration, not once per launch. In our runs, this case was always about 3% slower than the prefetched one. We will not dig into that here.
* **Prefetch.** Moving the same 512 MB with `cudaMemPrefetchAsync` takes 26 ms, and then the kernel runs at full speed. That is five times faster than letting the faults do it. It is even twice as fast as the pageable `cudaMemcpy` of the baseline, most likely because the driver copies the pages directly, without the extra staging copy of pageable memory that we saw in post 12.
* **Stays on CPU.** With the advice, the kernel reads all its data over PCIe: 768 MB in 51 ms. That beats the page faults for a single pass, but the data never moves, so every future launch would pay the same price again.

Your numbers will depend on your PCIe generation and CPU memory, but the order of the cases should be the same.

---

## 7. Seeing the Page Faults in `nsys`

In post 04, we used `nsys` to see copies and kernels on a timeline. It can also record unified memory activity. That is off by default because it slows the program down, so we turn it on with two flags. Let's profile only case 2, the page-fault case:

```bash
nsys profile -t cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
    -o faults ./unified-memory 2
nsys stats -q --report um_total_sum,cuda_gpu_mem_size_sum faults.nsys-rep
```

```text
 ** Unified Memory Totals Summary (um_total_sum):

 Total HtoD Migration Size (MB)  Total DtoH Migration Size (MB)  Total CPU Page Faults  Total GPU PageFaults  Minimum Virtual Address  Maximum Virtual Address
 ------------------------------  ------------------------------  ---------------------  --------------------  -----------------------  -----------------------
                        314.069                         268.370                  2,303                 2,484  0x7AA608000000           0x7AA637F00000


 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation
 ----------  ------  --------  --------  --------  --------  -----------  ------------------------------------
    314.069  10,289     0.031     0.008     0.004     1.032        0.109  [CUDA memcpy Unified Host-to-Device]
    268.370   1,534     0.175     0.033     0.004     1.044        0.301  [CUDA memcpy Unified Device-to-Host]
```

The GPU raised 2,484 page faults. The host-to-device migrations happened in 10,289 separate copies, with a median size of just 8 KB. That is the cost we measured: thousands of tiny transfers, each one after a fault round trip. The device-to-host part (268 MB) is `c` moving back to the CPU when `count_errors` reads it, and the CPU page faults come from the CPU touching pages in `fill` and `count_errors`.

We expected 512 MB host-to-device for `a` and `b`, but `nsys` reports less, and the number changes from run to run (we saw between 253 and 374 MB). We could not find out why `nsys` misses part of the migrations. The data did move: the second launch runs at full speed, which it could not do if half of `a` and `b` were still on the CPU. So trust the counts and the sizes of the pieces here more than the total.

Now the same for case 3, the prefetch:

```bash
nsys profile -t cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
    -o prefetch ./unified-memory 3
nsys stats -q --report cuda_gpu_mem_size_sum prefetch.nsys-rep
```

```text
 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
    536.871    256     2.097     2.097     2.097     2.097        0.000  [CUDA memcpy Unified Host-to-Device]
    268.435  1,536     0.175     0.033     0.004     1.044        0.301  [CUDA memcpy Unified Device-to-Host]
```

Exactly 512 MiB (536.871 MB) in 256 pieces of 2 MB each, and no GPU page faults at all, so `nsys` has nothing to print for `um_total_sum`. That is what a prefetch looks like. If you open the report in the Nsight Systems GUI (`nsys-ui faults.nsys-rep`), the faults and migrations also appear as their own rows under the GPU on the timeline.

---

## 8. Convenient or Costly?

Unified memory is a good fit when:

* **You are writing a first version.** One pointer and no copies make the code short. Get it correct first, then add prefetches where the profiler shows faults.
* **The data is full of pointers.** A tree or a linked list built on the CPU with `cudaMallocManaged` nodes can be walked by the GPU as it is. With `cudaMalloc`, every pointer would have to be translated.
* **The GPU only touches a small part of a large array.** With faults, only the pages that are really used move over PCIe. A `cudaMemcpy` would copy everything.
* **The data does not fit in GPU memory.** Managed allocations can be bigger than VRAM; the driver evicts pages back to the CPU to make room. `cudaMalloc` would simply fail. It is slow, but it works.

It silently hurts when:

* **The CPU touches the data first and the kernel faults it over.** This is our 160x case, and it is the default thing that happens when you port code by replacing `malloc` with `cudaMallocManaged`.
* **The CPU and the GPU take turns on the same data.** Every time the CPU reads a page, it moves back, and the next kernel faults on it again. The exercise below shows this.
* **You time a kernel without knowing it.** A kernel that suddenly takes 100 ms may be fine; it is just paying for migrations. `nsys` with the UM flags tells you which one it is.
* **You are on Windows.** Without `concurrentManagedAccess`, every launch moves all managed memory to the GPU, and the CPU cannot touch it while kernels run.

The rule of thumb: use unified memory for convenience, but for data you know the kernel needs, add a `cudaMemPrefetchAsync` before the launch. It is one line, and it turns the worst case into the best one.

---

## 9. Exercise

In case 3, after the kernel and `count_errors`, launch the kernel a second time and time it. `count_errors` reads all of `c` on the CPU, so `c` moves back to CPU memory, and the second kernel has to fault it back again. On our RTX 4090, the second kernel went from 0.83 ms to about 50 ms.

Then fix it: add `cudaMemPrefetchAsync(c.get(), bytes, gpu, 0, 0)` before the second launch and time the prefetch and the kernel separately. We got about 12 ms for the prefetch and a normal 0.85 ms kernel. Run both versions under `nsys` with the UM flags and compare the number of GPU page faults.

Bonus: the kernel overwrites `c` without reading it, so moving the old `c` back is wasted work. Look at `cudaMemDiscardAndPrefetchBatchAsync` in `cuda_runtime_api.h`, a recent addition that tells the driver the old contents are no longer useful, so it can prefetch without copying them.

In summary, `cudaMallocManaged` gives us one pointer for the CPU and the GPU, and the driver moves pages to whoever touches them. On Linux, that happens through GPU page faults, which are correct but slow: our vector add went from 0.84 ms to 135 ms when the data was still on the CPU. `cudaMemPrefetchAsync`, which takes a `cudaMemLocation` in CUDA 13, moves the data ahead of time in large pieces and gives us the full speed back, and `cudaMemAdvise` sets longer-term policies like where the data should live. `nsys` with `--cuda-um-gpu-page-faults=true` shows exactly how many faults and migrations happened. In the next post, we will look at CUDA Graphs, which cut the cost of launching many small kernels.

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming "CUDA Programming Guide: Unified Memory Programming"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html "CUDA Runtime API: Memory Management"
[3]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "Nsight Systems User Guide"
[4]: https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/ "Maximizing Unified Memory Performance in CUDA"
