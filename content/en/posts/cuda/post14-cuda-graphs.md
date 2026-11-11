---
title: "Post 14: CUDA Graphs"
date: 2026-09-29T19:14:00+03:00
weight: 15
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 12, we used streams to keep the GPU busy: while one chunk was being copied, another one was being computed. That works well when each piece of work is large. But many real programs are not like that. A simulation step, a small neural network layer, or an image pipeline often runs dozens of *tiny* kernels, one after another, over and over again. Each kernel finishes in about a microsecond, and suddenly the cost of *launching* it matters more than the work it does.

In this post, we will measure that launch cost and then remove most of it with CUDA Graphs. The idea is simple: instead of asking the driver to launch 20 kernels one by one, we record the 20 launches once, turn them into a *graph*, and then replay the whole graph with a single call.

## What We Will Learn

In this article, we will cover:

* Why launching many tiny kernels is slow, and how to measure the cost per kernel.
* How to record kernel launches into a graph with stream capture.
* How to instantiate a graph with `cudaGraphInstantiate` and replay it with `cudaGraphLaunch`.
* How to change a kernel argument in an existing graph without rebuilding it.
* How to free graph objects with `unique_ptr`.

## Table of Contents

1. The Full Program.
2. The Workload.
3. The Baseline: Launching Kernels One by One.
4. Recording a Graph.
5. Replaying the Graph.
6. Updating a Graph.
7. Build and Run.
8. What `nsys` Shows.
9. Building a Graph by Hand.
10. Exercise.

---

## 1. The Full Program

```bash
mkdir cuda-graphs
cd cuda-graphs
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post14/main.cu" >}}

We keep `CUDA_CHECK` and `DeviceBuffer` from [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}). Let's go through the new parts.

---

## 2. The Workload

{{< code file="cuda/post14/main.cu" region="kernel" link="false" >}}

The kernel is as small as it gets: every thread adds `a` to one element. We run it on only 16,384 floats (64 blocks of 256 threads), so the GPU has almost nothing to do. That is on purpose. We want a kernel whose run time is tiny, so that anything else, like the launch itself, becomes visible.

{{< code file="cuda/post14/main.cu" region="setup" link="false" >}}

One *iteration* of our workload is 20 launches of `step` in a row, and we run 1000 iterations. That is 20,000 kernel launches in total. With `a = 1`, every element should end up equal to 20,000, which is easy to check. The lambda `enqueue_iteration` puts one iteration into our stream. We will use it for both versions of the program, so they run exactly the same kernels.

We create our own stream instead of using the default one. Stream capture, which we will use in a moment, does not work on the legacy default stream.

The stream, the events, and the graph objects are all handles: `cudaStream_t` is really a pointer to an opaque struct called `CUstream_st`, and the same goes for the others. So we can give them to a `unique_ptr` with their destroy function as the deleter, just like we did with `cudaFree` for device memory and with the stream in post 12:

{{< code file="cuda/post14/main.cu" region="handles" link="false" >}}

---

## 3. The Baseline: Launching Kernels One by One

{{< code file="cuda/post14/main.cu" region="baseline" link="false" >}}

We run one iteration first as a warm-up. The very first launch of a kernel loads its code onto the GPU, and we do not want to measure that. Then we reset the buffer to zero with `cudaMemsetAsync` and time the 1000 iterations.

We measure two different things here:

* The **GPU time**, with a pair of `cudaEvent_t` like in post 03. This is how long the GPU took from the start event to the stop event.
* The **CPU submit time**, with `std::chrono`. This is how long the CPU needed just to *queue* the 20,000 launches, before we wait for anything.

Remember that a kernel launch is asynchronous: the CPU puts a command in the stream and moves on. If the CPU can queue commands faster than the GPU runs them, the CPU time will be much smaller than the GPU time. If the two numbers are the same, the GPU was waiting for the CPU.

---

## 4. Recording a Graph

{{< code file="cuda/post14/main.cu" region="capture" link="false" >}}

A CUDA graph is a list of operations (kernels, copies, memsets) plus the dependencies between them. The easiest way to build one is *stream capture*:

1. `cudaStreamBeginCapture` puts the stream into capture mode.
2. Everything we enqueue into the stream is now **recorded, not executed**. Our 20 kernel launches return immediately, and the GPU does nothing.
3. `cudaStreamEndCapture` stops recording and gives us a `cudaGraph_t`.

The nice thing about capture is that we did not change our code at all. The same `enqueue_iteration` that launched kernels in the baseline now builds a graph. Since all 20 launches went into one stream, the graph knows that each kernel depends on the one before it.

`cudaStreamCaptureModeGlobal` is the safest mode. It makes some unsafe calls, like a `cudaMalloc` from another thread during the capture, fail loudly instead of silently misbehaving.

A `cudaGraph_t` is only a description. Before we can run it, we turn it into an *executable graph* with `cudaGraphInstantiate`. This is where the driver does the expensive work once: it checks the graph, prepares the launch parameters for every node, and uploads what it can to the GPU.

If you read older tutorials, you may see `cudaGraphInstantiate` with five arguments (an error node, a log buffer, and its size). That form was removed in CUDA 12. In CUDA 13, the signature is:

```cpp
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                 cudaGraph_t graph,
                                 unsigned long long flags = 0);
```

We time the capture and the instantiation together with `std::chrono`, because this is a one-time cost that we pay before the first replay.

---

## 5. Replaying the Graph

{{< code file="cuda/post14/main.cu" region="replay" link="false" >}}

`cudaGraphLaunch` runs the whole graph, all 20 kernels, in the stream. We warm up once, reset the buffer, and then do exactly the same 1000 iterations as the baseline. The only difference is that the CPU now makes 1000 calls instead of 20,000.

At the end of `main`, the `unique_ptr`s call `cudaGraphExecDestroy` and `cudaGraphDestroy` for us. Note that you do not need to keep the `cudaGraph_t` alive to launch the executable graph. We keep it only because we already have it; you could destroy it right after instantiating.

---

## 6. Updating a Graph

A graph records the kernel arguments too. When we captured it, `a` was `1.0f`, and every replay will use `1.0f`. What if the next step of our program needs a different value?

Instantiating a new graph costs about 150 µs, as we will see below. That is fine once, but not every iteration. CUDA lets us patch an existing executable graph instead:

{{< code file="cuda/post14/main.cu" region="update" link="false" >}}

We capture the same 20 launches again, this time with `a = 2.0f`, and call `cudaGraphExecUpdate`. It compares the new graph with the one the executable was made from. If they have the same *shape* (the same nodes, the same kernels, the same dependencies) and only parameters changed, it copies the new parameters into the existing executable graph. That is much cheaper than a new `cudaGraphInstantiate`.

If the shape changed, for example because we captured 21 kernels instead of 20, the update fails and `info.result` tells us why. In that case, the usual approach is to destroy the old executable graph and instantiate a new one.

We check the result by running the updated graph once on a zeroed buffer: every element must be 20 × 2 = 40.

---

## 7. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -O3 -arch=native main.cu -o cuda-graphs
./cuda-graphs
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post14/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/cuda-graphs
```
{{< /tab >}}
{{< /tabs >}}

Output: (RTX 4090)

```text
1000 iterations x 20 kernels = 20000 launches, 16384 elements each
                  GPU time   per kernel   CPU submit
stream launches :  33.33 ms    1.67 us    33.32 ms  PASSED
graph replay    :  18.54 ms    0.93 us     1.48 ms  PASSED
capture + instantiate (once): 153.2 us
speedup: 1.80x
graph update with a = 2: PASSED
```

This is how the program prints its report:

{{< code file="cuda/post14/main.cu" region="report" link="false" >}}

Let's read the numbers.

**Stream launches: 1.67 µs per kernel.** Look at the last column: the CPU needed 33.32 ms to queue the launches, and the GPU finished at 33.33 ms. The two numbers are the same. That means the GPU was never ahead of work; it ran each kernel as soon as the CPU managed to submit it, then sat idle until the next one arrived. Our program is **CPU-bound**: the real limit is the roughly 1.7 µs the CPU and the driver need for each `<<<...>>>` launch.

**Graph replay: 0.93 µs per kernel.** Now the CPU queues all 1000 graph launches in only 1.48 ms, so the GPU always has work waiting. The GPU becomes the limit, and it runs the kernels back to back. The same 20,000 kernels take 18.54 ms instead of 33.33 ms, which is 1.8 times faster, without changing a single line of the kernel.

**Capture + instantiate: about 150 µs.** This is a one-time cost. We save about 0.74 µs per kernel, or about 15 µs per iteration of 20 kernels, so the graph pays for itself after about ten iterations. In our three runs, the CPU submit time for the graph was 1.5 to 2.2 ms, and the speedup was between 1.79x and 1.89x.

One more detail: the very first run after building took 7 ms instead of 150 µs for capture and instantiate. The following runs were back to around 150 µs. Treat the first run of a fresh binary as a warm-up, and do not base conclusions on it.

The CPU submit column also tells us something important: the CPU is now almost free. In a real program, those 30 ms of CPU time could be spent preparing the next batch of data.

---

## 8. What `nsys` Shows

We can confirm this story with Nsight Systems from post 04:

```bash
nsys profile -o graphs ./cuda-graphs
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum graphs.nsys-rep
```

Here are the interesting lines (trimmed):

```text
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------------------------
    100.0       20,220,244     20,020   1,010.0   1,024.0       992     1,760         16.8  step(float *, int, float)

 ** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)               Name
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  -----------------------------
     26.0       50,761,371     20,060        2,530.5        2,415.0          551       53,129         745.4  cudaLaunchKernel
      2.4        4,778,307      1,002        4,768.8        4,579.0        4,238       23,794       1,040.5  cudaGraphLaunch_v10000
      0.1          112,370          1      112,370.0      112,370.0      112,370      112,370           0.0  cudaGraphInstantiate_v12000
      0.0            5,961          1        5,961.0        5,961.0        5,961        5,961           0.0  cudaGraphExecUpdate_v10020
```

A few things stand out:

* Each `step` kernel runs for about **1.0 µs** on the GPU. The graph replay reached 0.93 µs per kernel, about the same as the kernel's own run time, so there are almost no gaps left between kernels.
* One `cudaLaunchKernel` call costs about 2.4 µs of CPU time here, while one `cudaGraphLaunch` costs about 4.6 µs, but it launches 20 kernels. That is about 0.23 µs per kernel. Note that `nsys` adds some overhead to every API call it traces, which is why `cudaLaunchKernel` is slower here than the 1.7 µs we measured without the profiler.
* `cudaGraphExecUpdate` took 6 µs, compared to 112 µs for `cudaGraphInstantiate`.
* There are only 20,020 kernel instances: the 20 warm-up launches and the 20,000 baseline launches. By default, `nsys` traces a graph as one unit, not each kernel inside it. Add `--cuda-graph-trace=node` to `nsys profile` if you want to see the kernels inside the graphs too.

---

## 9. Building a Graph by Hand

Stream capture is not the only way to make a graph. You can also build one node by node with the *explicit* graph API. Here are the same 20 kernels:

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

void* args[] = {&x, &n, &a};
cudaKernelNodeParams p{};
p.func = reinterpret_cast<void*>(step);
p.gridDim = dim3(blocks);
p.blockDim = dim3(threads);
p.kernelParams = args;

cudaGraphNode_t prev = nullptr;
for (int k = 0; k < 20; ++k) {
    cudaGraphNode_t node;
    cudaGraphAddKernelNode(&node, graph, prev ? &prev : nullptr,
                           prev ? 1 : 0, &p);
    prev = node;
}
```

Each node lists the nodes it depends on. Here, each kernel depends on the previous one, which gives the same chain we got from the stream. This is more code, but it gives you full control over the dependencies, and you keep a `cudaGraphNode_t` handle for every node. With that handle, `cudaGraphExecKernelNodeSetParams(exec, node, &p)` changes the arguments of one single kernel in an executable graph, without capturing anything again.

In practice, stream capture is what most code uses, because it also records library calls (like cuBLAS or CUB) that you could not easily write as nodes yourself. Use the explicit API when you want to build the dependency graph from your own data, or when you need to patch individual nodes often.

---

## 10. Exercise

Change `n` to `1 << 22` (about 4 million elements) and run the program again. What happens to the speedup, and why?

Here is what we got:

```text
1000 iterations x 20 kernels = 20000 launches, 4194304 elements each
                  GPU time   per kernel   CPU submit
stream launches : 205.34 ms   10.27 us   195.02 ms  PASSED
graph replay    : 182.70 ms    9.13 us     1.50 ms  PASSED
capture + instantiate (once): 205.7 us
speedup: 1.12x
graph update with a = 2: PASSED
```

Hint: each kernel now takes about 9 µs on the GPU, far more than the 1.7 µs it takes to launch it. Who is waiting for whom now? And why is the CPU submit time of the baseline almost as long as the GPU time, even though the CPU is no longer the bottleneck? (The stream can only hold a limited number of pending launches. When it is full, the next launch blocks until the GPU makes room.)

As a second experiment, keep the small `n`, and change `kernels_per_iter` and `iters` to 1 × 20,000 and to 200 × 100, so the total stays at 20,000 kernels. How does the size of the graph change the result?

In summary, every kernel launch costs a few microseconds of CPU and driver time. For large kernels, that cost is hidden, but for kernels that run in about a microsecond, the launch becomes the bottleneck and the GPU sits idle. CUDA Graphs fix this by recording a sequence of launches once, with `cudaStreamBeginCapture` and `cudaStreamEndCapture`, turning it into an executable graph with `cudaGraphInstantiate`, and replaying it with a single `cudaGraphLaunch`. On our RTX 4090, that took 20,000 tiny kernels from 1.67 µs to 0.93 µs each, and cut the CPU time from 33 ms to 1.5 ms. When only the arguments change, `cudaGraphExecUpdate` patches the graph for a fraction of the cost of instantiating a new one. And like any other CUDA handle, graphs and executable graphs can be owned by a `unique_ptr` so they are always destroyed.

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs "CUDA Programming Guide: CUDA Graphs"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html "CUDA Runtime API: Graph Management"
[3]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html "CUDA Runtime API: Stream Management"
[4]: https://developer.nvidia.com/blog/cuda-graphs/ "Getting Started with CUDA Graphs"
[5]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "Nsight Systems User Guide"
