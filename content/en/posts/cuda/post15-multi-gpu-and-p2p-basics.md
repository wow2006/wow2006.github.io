---
title: "Post 15: Multi-GPU & P2P basics"
date: 2026-09-29T19:15:00+03:00
weight: 16
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

This post is *optional*. Most readers have one GPU, and nothing later in the series depends on it. If you have two, read on.

In post 12, we used pinned memory and streams to keep one GPU and its PCIe link busy at the same time. The machine I am writing this series on has two RTX 4090s, so the natural next question is: can we split the work across both GPUs and finish in half the time? And can the two GPUs copy data directly to each other, without going through the CPU?

We will answer both questions with real numbers. The short answers are "partly" and "not on this card".

## What We Will Learn

In this article, we will cover:

* How to count the GPUs with `cudaGetDeviceCount` and pick one with `cudaSetDevice`.
* How to split a vector operation across two GPUs, with one set of buffers and one stream per GPU.
* How to check for peer access with `cudaDeviceCanAccessPeer`, and enable it with `cudaDeviceEnablePeerAccess`.
* How to copy between GPUs with `cudaMemcpyPeer`, and how fast that is on GeForce cards.
* Why the "nv-p2p deprecation" in CUDA 13.0 is not about any of the above.

## Table of Contents

1. The Machine.
2. The Full Program.
3. Counting and Selecting GPUs.
4. Splitting the Work.
5. Peer Access.
6. Copying Between GPUs.
7. What About the nv-p2p Deprecation?
8. Build and Run.
9. Exercise.

---

## 1. The Machine

Before writing any code, let's ask the driver what we have:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv
nvidia-smi topo -m
```

```text
GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NODE	0-63	0		N/A
GPU1	NODE	 X 	0-63	0		N/A
```

`NODE` means the two GPUs are connected only through PCIe and the CPU (an AMD Threadripper PRO here). There is no NVLink between them. Each GPU has its own PCIe 4.0 x16 link to the CPU.

`nvidia-smi` can also tell us whether the GPUs can talk to each other directly:

```bash
nvidia-smi topo -p2p r
```

```text
 	GPU0	GPU1
 GPU0	X	GNS
 GPU1	GNS	X

Legend:
  ...
  GNS  = GPU not supported
```

`GNS` means "GPU not supported". NVIDIA does not enable peer-to-peer (P2P) access on GeForce RTX 40-series cards; it is kept for the professional and data-center cards. We will see the same answer from the CUDA API in section 5. Keep this in mind: the numbers in this post are what you get on two consumer cards, not on a DGX.

---

## 2. The Full Program

```bash
mkdir multi-gpu
cd multi-gpu
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post15/main.cu" >}}

The kernel is the same `add` from post 01. Everything new is on the host side: which GPU we talk to, and where the data goes.

---

## 3. Counting and Selecting GPUs

{{< code file="cuda/post15/main.cu" region="count" link="false" >}}

`cudaGetDeviceCount` returns the number of GPUs the program can see. They are numbered from 0. We already used `cudaGetDeviceProperties` in post 00; here we also print the PCI bus ID, so we can match each CUDA device to a line in `nvidia-smi`.

The key idea of multi-GPU CUDA is the *current device*. Every host thread has one, and it starts as device 0. `cudaSetDevice(dev)` changes it. After that, almost every call goes to that GPU:

* `cudaMalloc` allocates on the current device.
* `cudaStreamCreate` creates a stream on the current device.
* A kernel launch runs on the current device.

So the pattern for two GPUs is always the same: `cudaSetDevice(0)`, do the GPU 0 work, `cudaSetDevice(1)`, do the GPU 1 work.

Our buffer helpers are the same `unique_ptr` wrappers as before. The comments are the only reminder we need that they allocate on whatever device is current:

{{< code file="cuda/post15/main.cu" region="buffers" link="false" >}}

Freeing is simpler. Since CUDA 4, all GPUs and the host share one virtual address space (*unified virtual addressing*), so the driver knows which GPU a pointer belongs to. That is why `cudaFree` and `cudaStreamDestroy` in our deleters work no matter which device is current when the `unique_ptr` goes out of scope.

The CUDA device numbers can be changed from outside the program with the `CUDA_VISIBLE_DEVICES` environment variable. This is useful for testing the one-GPU path on a two-GPU machine:

```bash
CUDA_VISIBLE_DEVICES=1 ./multi-gpu
```

```text
Found 1 GPU(s)
  GPU 0: NVIDIA GeForce RTX 4090, PCI bus 61
1 GPU(s):  34.50 ms per run  PASSED (0 errors)
Only one GPU, skipping the peer-to-peer part
```

The program now sees one GPU, and calls it GPU 0, even though it is the card on bus 61.

---

## 4. Splitting the Work

We add two vectors of 64M floats (256 MiB each), like in post 01, but this time the data starts and ends in host memory, and we time the whole job: copy in, add, copy out. The host arrays are pinned, as we learned in post 12:

{{< code file="cuda/post15/main.cu" region="host" link="false" >}}

We use `unique_ptr<float[]>` here so that we can write `a[i]` on the host. We do not need to tell `cudaMallocHost` which GPU the memory is for. With unified virtual addressing, pinned memory is usable by every GPU.

To split the work, each GPU gets a *slice*: an offset into the host arrays, a count, three device buffers of its own, and a stream of its own.

{{< code file="cuda/post15/main.cu" region="slices" link="false" >}}

With two GPUs, GPU 0 takes elements 0 to 32M−1 and GPU 1 takes the rest. The buffers are created right after `cudaSetDevice(dev)`, so each slice's memory lives on its own GPU. A GPU cannot read another GPU's memory here (section 5 shows why), so every GPU must have its own copy of the data it works on.

Now we queue the work:

{{< code file="cuda/post15/main.cu" region="launch" link="false" >}}

The first loop does not wait for anything. `cudaMemcpyAsync` and the kernel launch only put work into a stream and return, so the loop finishes in microseconds, and both GPUs are busy at the same time. Only the second loop waits, once per GPU.

This is why the per-device streams matter. If we used the blocking `cudaMemcpy` instead, the CPU would wait for GPU 0's copy to finish before it even started GPU 1's, and the two GPUs would take turns instead of working together.

We time this with `std::chrono` instead of the `cudaEvent_t` timer from post 03. An event belongs to one device, and we want the wall time from "start queuing on GPU 0" to "both GPUs done". Since we synchronize with both GPUs before reading the clock, a host timer is correct here.

Finally, we run the same job on one GPU and then on two, and check every element each time:

{{< code file="cuda/post15/main.cu" region="compare" link="false" >}}

Output:

```text
1 GPU(s):  34.93 ms per run  PASSED (0 errors)
2 GPU(s):  26.47 ms per run  PASSED (0 errors)
```

Two GPUs are only 1.3 times faster, not 2 times. Why?

The job is almost all copying. Each run moves 768 MiB across PCIe (two arrays in, one out), and the kernel is tiny. Looking at the run in `nsys` (post 04) makes it clear:

| | 1 GPU | 2 GPUs (each) |
| --- | --- | --- |
| `add` kernel | 0.82 ms | 0.41 ms |
| Host-to-device copy speed | ~22 GB/s | ~15 GB/s |

The kernel time is split exactly in half, as we hoped. But the copies are not. Alone, one GPU copies at about 22 GB/s, the same as in post 12. When both GPUs copy at the same time, each one only gets about 15 GB/s. Together that is about 30 GB/s, more than one link, but not twice as much. The two PCIe links are separate, but they end in the same CPU and the same host memory system, and on this machine that side is the limit.

So splitting across GPUs pays off in full for the work that stays *on* the GPU. For work that is limited by the host, like this one, you get what the host can deliver. A real multi-GPU program copies the data in once and then does a lot of work on it.

---

## 5. Peer Access

Peer access means one GPU can read and write another GPU's memory directly: a kernel on GPU 0 dereferences a pointer to GPU 1's memory, and the data crosses PCIe (or NVLink) without touching host memory. It is a two-step API. First we ask whether it is possible, then we turn it on:

{{< code file="cuda/post15/main.cu" region="can-access" link="false" >}}

`cudaDeviceCanAccessPeer(&can, a, b)` sets `can` to 1 if device `a` can access the memory of device `b`. Access is one-directional, so we check both ways. If both are possible, `cudaDeviceEnablePeerAccess(peer, 0)` lets the *current* device access `peer`'s memory. The last argument is a flags value that must be 0.

On this machine, the answer is no, in both directions:

```text
Can GPU 0 access GPU 1 memory? no
Can GPU 1 access GPU 0 memory? no
```

This is the same `GNS` we saw in `nvidia-smi topo -p2p r`. If you skip the check and call `cudaDeviceEnablePeerAccess` anyway, it fails with:

```text
cudaErrorPeerAccessUnsupported: peer access is not supported between these two devices
```

That is why the program only enables peer access when both checks pass. On cards where P2P works (for example, data-center GPUs, or professional cards connected with NVLink), the same code enables it, and the copy in the next section goes straight from GPU to GPU.

---

## 6. Copying Between GPUs

`cudaMemcpyPeer` copies from one GPU's memory to another's. It takes the device number of each side:

{{< code file="cuda/post15/main.cu" region="peer-copy" link="false" >}}

The argument order is the one we know from `cudaMemcpy`: destination first. `cudaMemcpyPeer` does not need peer access to be enabled. If the GPUs cannot talk directly, the driver copies through host memory for us: from GPU 0 into a staging buffer on the host, then from there to GPU 1.

To see how good the driver's version is, we also do the same thing by hand with two plain copies through our pinned buffer:

{{< code file="cuda/post15/main.cu" region="staged-copy" link="false" >}}

And we check that the peer copy really arrived:

{{< code file="cuda/post15/main.cu" region="verify-peer" link="false" >}}

Output:

```text
cudaMemcpyPeer GPU 0 -> GPU 1:  20.25 ms   13.3 GB/s  PASSED
Staged via host (D2H + H2D):    23.64 ms   11.4 GB/s
```

This number moves around between runs. Over several runs, `cudaMemcpyPeer` gave between 10.9 and 15.8 GB/s (most runs 13 to 15), and the manual version stayed at 11.3 to 11.9 GB/s.

The manual version is easy to explain. It copies 256 MiB out at about 24 GB/s, then 256 MiB in at about 23 GB/s, one after the other, so the total speed is about half of one copy.

`cudaMemcpyPeer` is a bit faster because the driver moves the data in smaller pieces and overlaps them: while one piece travels from the host to GPU 1, the next one is already coming out of GPU 0. It cannot reach the full 23 GB/s, though. Every byte crosses host memory twice (once in, once out), and we saw in section 4 that this machine delivers about 30 GB/s in total across both links. Half of that is about 15 GB/s, which is where the best runs land.

For comparison, in post 03 a `cudaMemcpy` inside one 4090's own memory ran at about 920 GB/s. Moving data between two GeForce cards is expensive. Design multi-GPU work on these cards so that each GPU needs as little of the other's data as possible.

---

## 7. What About the nv-p2p Deprecation?

If you read the CUDA 13.0 release notes, you will find this line [3]:

> The `nv-p2p` APIs are deprecated starting with CUDA 13.0 and will be removed in CUDA 14.0. These APIs remain functional in CUDA 13.0 but will no longer receive enhancements or bug fixes.

It is easy to read "p2p" and assume it is about `cudaMemcpyPeer` or `cudaDeviceEnablePeerAccess`. It is not.

`nv-p2p` is a header, `nv-p2p.h`, that ships with the NVIDIA *kernel driver*. It declares functions like `nvidia_p2p_get_pages` and `nvidia_p2p_dma_map_pages`, which are called from *Linux kernel modules*, not from CUDA programs. They exist for GPUDirect RDMA: letting a third-party device, such as a network card, read and write GPU memory directly [4]. The GPUDirect RDMA documentation points driver writers to the upstream Linux kernel APIs instead.

So:

| API | Who calls it | Status in CUDA 13 |
| --- | --- | --- |
| `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeer` | Your CUDA program | Current, not deprecated |
| `nvidia_p2p_*` in `nv-p2p.h` | Kernel drivers for GPUDirect RDMA | Deprecated in 13.0, removal planned for 14.0 |

Everything in this post uses the first row. Unless you write Linux drivers for network or storage hardware, the deprecation does not affect you.

---

## 8. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o multi-gpu
./multi-gpu
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post15/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/multi-gpu
```
{{< /tab >}}
{{< /tabs >}}

A note on `-arch=native` with two GPUs: it compiles for the GPUs in the machine. Here both are sm_89, so there is only one target. If your two GPUs are different models, `native` builds code for both.

Output:

```text
Found 2 GPU(s)
  GPU 0: NVIDIA GeForce RTX 4090, PCI bus 41
  GPU 1: NVIDIA GeForce RTX 4090, PCI bus 61
1 GPU(s):  34.93 ms per run  PASSED (0 errors)
2 GPU(s):  26.47 ms per run  PASSED (0 errors)
Can GPU 0 access GPU 1 memory? no
Can GPU 1 access GPU 0 memory? no
cudaMemcpyPeer GPU 0 -> GPU 1:  20.25 ms   13.3 GB/s  PASSED
Staged via host (D2H + H2D):    23.64 ms   11.4 GB/s
```

If you have a card with P2P support, you will also see `Peer access enabled in both directions`, and the `cudaMemcpyPeer` line should be much faster than the staged one. Running the program under `compute-sanitizer` reports `ERROR SUMMARY: 0 errors`, as usual.

---

## 9. Exercise

We split the vector evenly because our two GPUs are the same. Let's see what happens when the split is not even. In `split_add`, replace the line that computes `per_gpu` with:

```cpp
int per_gpu = ngpu == 1 ? n : n / 4 * 3;
```

Now GPU 0 gets three quarters of the elements and GPU 1 gets one quarter. Before you run it, guess the time for two GPUs.

We measured:

```text
1 GPU(s):  36.28 ms per run  PASSED (0 errors)
2 GPU(s):  33.34 ms per run  PASSED (0 errors)
```

Almost all of the gain is gone. GPU 1 finishes early and then waits, while GPU 0 still has most of the job. The total time is set by the slowest GPU, which is why an even split, or a split proportional to each GPU's speed if they are different models, matters.

As a second experiment, open `nsys` on the original program and look at the two GPU rows. You should see the copies on GPU 0 and GPU 1 running side by side, each slower than when one GPU runs alone.

In summary, multi-GPU CUDA starts with one idea: `cudaSetDevice` picks the GPU that the next allocations, streams, and launches go to. To use two GPUs at once, give each one its own slice, its own buffers, and its own stream, queue everything asynchronously, and only then wait. The work on the GPUs scaled perfectly here, but the copies were limited by the host, so the total was 1.3 times faster. `cudaDeviceCanAccessPeer` tells you whether direct GPU-to-GPU access is possible; on GeForce RTX 40-series cards it is not, and `cudaMemcpyPeer` falls back to going through the host at 11 to 16 GB/s. Finally, the `nv-p2p` deprecation in CUDA 13.0 is about a kernel-driver API for GPUDirect RDMA, not about the runtime peer-access API we used.

[1]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html "CUDA Runtime API: Device Management"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html "CUDA Runtime API: Peer Device Memory Access"
[3]: https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html "CUDA 13.0 Update 1 Release Notes"
[4]: https://docs.nvidia.com/cuda/gpudirect-rdma/index.html "GPUDirect RDMA"
