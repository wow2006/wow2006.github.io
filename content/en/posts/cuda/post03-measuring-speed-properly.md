---
title: "Post 03: Measuring speed properly"
date: 2026-09-29T19:03:00+03:00
weight: 4
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}), we added two vectors on the GPU and checked that the answer was correct. We never asked how fast it was. That is the whole point of using a GPU, so from now on, every kernel in this series will come with a number.

Getting that number right is harder than it looks. A kernel launch is asynchronous, so the obvious approach of wrapping it in `std::chrono` measures the wrong thing. The first launch is slower than the rest. And a time in milliseconds alone does not tell us if the kernel is good or bad. In this post, we will build a small timing helper, turn its result into GB/s and GFLOP/s, and compare that with what the hardware can do at most. That comparison is the "score" we will use in every later post.

## What We Will Learn

In this article, we will cover:

* Why a CPU timer lies about kernel time.
* How to time a kernel with `cudaEvent_t`.
* Why we need warm-up runs and averaging.
* How to compute effective bandwidth (GB/s) and compute (GFLOP/s).
* How to query the GPU's theoretical peak, and what CUDA 13 changed there.
* How to tell if a kernel is memory-bound or compute-bound, with a first look at the roofline model.

## Table of Contents

1. The Full Program.
2. Why CPU Timers Lie.
3. Timing with CUDA Events.
4. Warm-up and Averaging.
5. From Milliseconds to a Score.
6. The Theoretical Peak.
7. A Light Roofline.
8. Build and Run.
9. Exercise.

---

## 1. The Full Program

```bash
mkdir measure-speed
cd measure-speed
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post03/main.cu" >}}

The `add` kernel, `CUDA_CHECK`, and `make_device_buffer` are the same as in the previous posts. This time we use 32M floats (128 MB per buffer) instead of one million. We will see why the size matters at the end of the post. We also skip the host vectors and just zero the inputs with `cudaMemset`, because here we only care about speed.

---

## 2. Why CPU Timers Lie

The first idea most of us have is to time the kernel like any other C++ function:

{{< code file="cuda/post03/main.cu" region="chrono" link="false" >}}

`run_add` is a small lambda that launches the kernel, so we do not have to repeat the launch line every time:

{{< code file="cuda/post03/main.cu" region="work" link="false" >}}

Here is what the two timers print on an RTX 4090:

```text
std::chrono, no sync:          0.003 ms
std::chrono, with sync:        0.441 ms
```

The first number says the kernel took 3 microseconds. It did not. As we saw in [post 00]({{< ref "/posts/cuda/post00-helloworld-cuda" >}}), a launch is asynchronous: the CPU puts the kernel in a queue and returns immediately. So `t1 - t0` is only the cost of *queueing* the launch. The GPU has barely started.

Adding `cudaDeviceSynchronize` fixes the worst of it, because now the CPU waits for the GPU to finish. But the timer still includes things that are not the kernel: the launch overhead, and the time it takes the CPU to notice that the GPU is done. For this kernel, that is a few percent. For a small kernel, it is most of the number. With one million elements instead of 32 million, the same code prints:

```text
std::chrono, with sync:        0.009 ms
cudaEvent, 3 warm-up + 20:     0.004 ms
```

The CPU timer says 9 microseconds; the kernel actually took 4. We need a clock that lives on the GPU.

---

## 3. Timing with CUDA Events

A CUDA event is a marker that we put in the GPU's queue. When the GPU reaches it, it writes down the time. If we record one event before the kernel and one after, the difference is exactly the GPU time between them, with no CPU in the way.

Here is the helper we will use for the rest of the series:

{{< code file="cuda/post03/main.cu" region="time_ms" link="false" >}}

Let's go through it:

* `cudaEventCreate` creates the two markers.
* `cudaEventRecord(start)` puts the first marker in the queue. Like a kernel launch, it returns immediately.
* We launch the work, then record `stop` behind it. Because everything goes into the same queue (the *default stream*), the GPU processes them in order: `start`, the kernels, then `stop`.
* `cudaEventSynchronize(stop)` makes the CPU wait until the GPU has reached `stop`. Before that, the stop time does not exist yet.
* `cudaEventElapsedTime` gives us the time between the two markers in milliseconds, with a resolution of about half a microsecond [1].

`work` can be anything callable: a lambda that launches a kernel, calls `cudaMemcpy`, or runs several kernels in a row. That is what makes the helper reusable.

We do not wrap each event call in `CUDA_CHECK`, because that macro returns `1` from the function, and this function returns a time. Instead, the caller checks right after:

{{< code file="cuda/post03/main.cu" region="score" link="false" >}}

`cudaGetLastError` returns the last error from *any* runtime call on this thread, not only from launches, so a failed event call or a crashed kernel still gets caught here.

---

## 4. Warm-up and Averaging

The helper has two parameters we have not explained yet: `warmup` and `reps`.

### Warm-up

The first time a kernel runs, it pays some one-time costs. Since CUDA 12.2, kernels are loaded onto the GPU *lazily* by default, the first time they are launched [2]. The GPU may also be in a low-power state and need a moment to raise its clocks. None of this is part of the kernel's real speed, so we run the work a few times before we start the clock.

Our program measures one cold launch on purpose, by calling the helper with no warm-up and a single repetition:

{{< code file="cuda/post03/main.cu" region="cold" link="false" >}}

With 32M elements, the cold launch takes 0.462 ms against 0.432 ms warm, about 7% slower. With one million elements, the difference is dramatic:

```text
First launch (cold):           0.084 ms
cudaEvent, 3 warm-up + 20:     0.004 ms
```

The first launch is 20 times slower than the others. If we had measured only once, we would have measured the setup, not the kernel.

### Averaging

A single run is noisy. Other programs share the GPU (on this machine, the desktop and a web browser run on the same card), and one run can land on a busy moment. So we launch the work `reps` times back to back between a single pair of events and divide by `reps`. This also makes the half-microsecond resolution of events irrelevant for short kernels.

Notice that we put one event pair around all 20 launches, not one pair around each launch. The launches are queued back to back, so the GPU never waits for the CPU between them. What we measure is the steady-state speed of the kernel.

---

## 5. From Milliseconds to a Score

0.432 ms tells us nothing on its own. Is that good? We need to compare it with something. For a kernel like `add`, the question to ask is: how many bytes did it move per second?

`add` reads `a` and `b` and writes `c`. That is three arrays of `n` floats:

```text
bytes = 3 * n * 4 = 3 * 33,554,432 * 4 = 402,653,184 bytes
```

Dividing bytes by seconds gives the *effective bandwidth*. "Effective" means we count the bytes the algorithm needs, not what the hardware actually transferred. The same idea works for compute: `add` does one floating-point addition per element, so it does `n` FLOPs (floating-point operations).

The last lines of the `score` region turn the time into both numbers and compare them with the peak of the GPU:

```text
add: 0.432 ms, 931 GB/s (92.4% of peak), 77.6 GFLOP/s (0.09% of peak)
```

This one line is the score format for the rest of the series: time, GB/s or GFLOP/s, and the percent of peak. Our humble vector add is using 92% of the memory bandwidth of the card, and 0.09% of its compute. Before we explain why, let's see where the peak numbers come from.

---

## 6. The Theoretical Peak

{{< code file="cuda/post03/main.cu" region="peak" link="false" >}}

`cudaDeviceGetAttribute` asks the driver for one property of the GPU at a time.

**Memory bandwidth.** The memory transfers data twice per clock (it is *double data rate*), and each transfer moves `bus_bits / 8` bytes. For the RTX 4090, the driver reports a memory clock of 10,501 MHz and a 384-bit bus:

```text
2 * 10,501,000,000 * 48 bytes = 1008 GB/s
```

which matches the 1008 GB/s on NVIDIA's spec sheet.

**Compute.** Each SM (streaming multiprocessor) has a fixed number of FP32 units, and each can do one fused multiply-add (`a * b + c`, counted as 2 FLOPs) per clock. The 4090 has 128 SMs with 128 units each, and a peak clock of 2520 MHz:

```text
2 * 128 * 128 * 2,520,000,000 = 82,575 GFLOP/s
```

There is no attribute for the number of FP32 units per SM, so we have to know it per architecture. It is 64 on Turing (compute capability 7.5) and on the A100 (8.0), and 128 on the consumer Ampere, Ada, and Blackwell cards, and on Hopper.

Treat both numbers as upper bounds, not promises. The GPU often boosts above the reported clock (we saw this one at 2715 MHz under load), and no real program reaches 100% of the memory bandwidth.

### What CUDA 13 changed

If you learned CUDA from older tutorials, you have probably seen the peak computed from `cudaDeviceProp`:

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);
double peak = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
```

With CUDA 13.1, this no longer compiles:

```text
error: class "cudaDeviceProp" has no member "memoryClockRate"
```

CUDA 13.0 removed several fields from `cudaDeviceProp`, including `clockRate` and `memoryClockRate` [3]. The values are still available through `cudaDeviceGetAttribute` with `cudaDevAttrClockRate` and `cudaDevAttrMemoryClockRate`, which is why we use attributes for everything above.

---

## 7. A Light Roofline

So why is `add` at 92% of the memory bandwidth but only 0.09% of the compute? Look at how much work it does per byte:

```text
arithmetic intensity = FLOPs / bytes = n / (12 * n) = 0.083 FLOP/byte
```

For every addition, it moves 12 bytes: two floats in, one float out. The GPU can do far more math than that per byte. If we divide its peak compute by its peak bandwidth, we get the intensity where the two limits meet, called the *ridge point*:

```text
ridge point = 82,575 GFLOP/s / 1008 GB/s = 81.9 FLOP/byte
```

{{< code file="cuda/post03/main.cu" region="roofline" link="false" >}}

This is the idea behind the *roofline model* [4]. The best speed a kernel can reach is:

```text
attainable GFLOP/s = min(peak GFLOP/s, intensity * peak GB/s)
```

* If a kernel's intensity is **below** the ridge point, memory is the limit. It is *memory-bound*, and its score is GB/s.
* If it is **above**, the math units are the limit. It is *compute-bound*, and its score is GFLOP/s.

For `add`, the roof is `0.083 * 1008 = 84 GFLOP/s`, and we measured 77.6 GFLOP/s. So the kernel is already close to the best it can ever be on this card. No amount of clever math will make it faster; only moving fewer bytes would. A kernel needs to do about 82 FLOPs per byte before the 4090's compute becomes the limit. Most simple kernels are far below that, which is why so many of the next posts are about memory.

---

## 8. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o measure-speed
./measure-speed
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post03/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/measure-speed
```
{{< /tab >}}
{{< /tabs >}}

Output: (RTX 4090)

```text
Peak memory bandwidth: 1008 GB/s
Peak FP32 compute:     82575 GFLOP/s

First launch (cold):           0.462 ms
std::chrono, no sync:          0.003 ms
std::chrono, with sync:        0.441 ms
cudaEvent, 3 warm-up + 20:     0.432 ms

add: 0.432 ms, 931 GB/s (92.4% of peak), 77.6 GFLOP/s (0.09% of peak)

Arithmetic intensity of add: 0.083 FLOP/byte
Ridge point of this GPU:     81.9 FLOP/byte
add is memory-bound
```

Your numbers will differ a little from run to run, and more if something else is using the GPU. That is normal. Run it a few times and look at the typical value, not the best one.

---

## 9. Exercise

Is 92% of peak good, or is our kernel wasting 8%? To find out, compare it with the fastest copy we can get. Add a copy kernel:

```cpp
__global__ void copy(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}
```

Then, at the end of `main`, time it and `cudaMemcpy` device-to-device with the same helper:

```cpp
double copy_bytes = 2.0 * n * sizeof(float);  // read in, write out
float copy_ms = time_ms([&] {
    copy<<<blocks, threads>>>(d_a.get(), d_c.get(), n);
});
float memcpy_ms = time_ms([&] {
    cudaMemcpy(d_c.get(), d_a.get(), n * sizeof(float), cudaMemcpyDeviceToDevice);
});
CUDA_CHECK(cudaGetLastError());
```

Print both scores in the same format as `add`. Note that a copy moves `2 * n * 4` bytes, not `3 * n * 4`: one read and one write per element.

On the RTX 4090, we got:

```text
copy:       0.291 ms, 923 GB/s (91.6% of peak)
cudaMemcpy: 0.291 ms, 921 GB/s (91.4% of peak)
```

Our five-line copy kernel is as fast as NVIDIA's own `cudaMemcpy`, and `add` is right there with them. About 92% is the practical ceiling for this card: DRAM spends some of its time on refresh and on switching between reading and writing. So `add` is not wasting anything.

Now change `n` to `1 << 20` (one million floats, 4 MB per buffer) and run again:

```text
add: 0.004 ms, 3236 GB/s (321.0% of peak), 269.7 GFLOP/s (0.33% of peak)
copy:       0.004 ms, 2308 GB/s (228.9% of peak)
cudaMemcpy: 0.003 ms, 2404 GB/s (238.5% of peak)
```

321% of peak is not a mistake in the formula. The 4090 has 72 MB of L2 cache, and our three 4 MB buffers fit in it. After the warm-up, the data never leaves the cache, so we are measuring the cache, not the memory. This is why we use 128 MB buffers when we want to measure memory bandwidth: always make your data larger than the L2 cache, or you will be measuring something else.

In summary, never time a kernel with a CPU timer alone: a launch returns before the kernel runs. Use `cudaEvent_t` pairs, warm up first, and average over many runs. Then turn the time into GB/s or GFLOP/s and compare it with the peak you get from `cudaDeviceGetAttribute`. The arithmetic intensity tells you which of the two is the right score. From now on, every kernel in this series will print that score, and our goal will be to push it toward the roof. In the next post, we will go beyond a single number and use Nsight Systems and Nsight Compute to see *why* a kernel is as fast as it is.

[1]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html "CUDA Runtime API: Event Management"
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading "CUDA Programming Guide: Lazy Loading"
[3]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html "CUDA Toolkit Release Notes"
[4]: https://en.wikipedia.org/wiki/Roofline_model "Roofline model"
