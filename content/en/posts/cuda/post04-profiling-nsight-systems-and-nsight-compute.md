---
title: "Post 04: Profiling with Nsight Systems and Nsight Compute"
date: 2026-09-29T19:04:00+03:00
weight: 5
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In the previous post, we timed kernels with `cudaEvent` and turned the time into GB/s. That tells us *how fast* a kernel is. It does not tell us *where* the time of the whole program goes, or *why* a kernel is as fast as it is. For that, we need a profiler.

CUDA comes with two, and they answer different questions:

* **Nsight Systems** (`nsys`) draws a timeline of the whole program: CPU calls, copies, kernels, and the gaps between them. Use it first, to find out what is slow.
* **Nsight Compute** (`ncu`) looks inside one kernel: how much of the memory bandwidth it uses, how many warps are running, and why they stall. Use it second, once you know which kernel matters.

If you learned CUDA from an older tutorial, you probably saw `nvprof` or the Visual Profiler. Both were removed in CUDA 13.0. They no longer exist in the toolkit, so any guide that starts with `nvprof ./a.out` is outdated. `nsys` and `ncu` replace them.

## What We Will Learn

In this article, we will cover:

* How to record a timeline with `nsys profile` and read it with `nsys stats`.
* How to mark parts of our code with NVTX ranges so they show up in the profile.
* How to profile one kernel in depth with `ncu`, and how to fix the performance counter permission error.
* How much of the vector add from post 01 is copying, and how much is computing.

## Table of Contents

1. The Program.
2. Build and Run.
3. A Timeline with Nsight Systems.
4. Reading the Timeline.
5. One Kernel in Depth with Nsight Compute.
6. The GUIs.
7. Exercise.

---

## 1. The Program

We profile the vector add from [post 01]({{< ref "/posts/cuda/post01-moving-data-between-cpu-and-gpu" >}}). The only change is a few NVTX calls around its three phases.

{{< code file="cuda/post04/main.cu" >}}

NVTX (NVIDIA Tools Extension) is a tiny API for putting named markers in your code. Version 3, which ships with the toolkit, is header-only:

{{< code file="cuda/post04/main.cu" region="include" link="false" >}}

There is nothing to link. When the program runs on its own, the calls do almost nothing. When it runs under `nsys`, each `nvtxRangePushA` / `nvtxRangePop` pair becomes a named bar on the timeline:

{{< code file="cuda/post04/main.cu" region="copy-in" link="false" >}}

We wrap the kernel in its own range too. Here we need `cudaDeviceSynchronize`, because the launch returns immediately. Without it, the `compute` range would end before the kernel even started:

{{< code file="cuda/post04/main.cu" region="launch" link="false" >}}

And the same for the copy back:

{{< code file="cuda/post04/main.cu" region="copy-out" link="false" >}}

---

## 2. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -O3 main.cu -o vector-add-nvtx
./vector-add-nvtx
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post04/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/vector-add-nvtx
```
{{< /tab >}}
{{< /tabs >}}

Output:

```text
Launched 3907 blocks x 256 threads for 1000000 elements
PASSED (0 errors)
```

Same as post 01. The profilers do not need a special build.

---

## 3. A Timeline with Nsight Systems

`nsys` is in `/usr/local/cuda/bin`, next to `nvcc`. Record a profile:

```bash
nsys profile -t cuda,nvtx -o vadd ./vector-add-nvtx
```

`-t cuda,nvtx` traces only CUDA calls and our NVTX ranges, which keeps the report small. `-o vadd` names the output file. The program runs normally, and `nsys` writes `vadd.nsys-rep`.

```text
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

Launched 3907 blocks x 256 threads for 1000000 elements
PASSED (0 errors)
Collecting data...
Generating '/tmp/nsys-report-b6ed.qdstrm'
Generated:
	/home/user/vector-add/vadd.nsys-rep
```

The two warnings are about CPU sampling, which Ubuntu restricts by default. We do not need it for GPU work, so we can ignore them.

The `.nsys-rep` file is binary. `nsys stats` turns it into tables. Let's start with our NVTX ranges:

```bash
nsys stats -q --report nvtx_sum vadd.nsys-rep
```

```text
 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style     Range
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------  ---------
     63.4          939,342          1  939,342.0  939,342.0   939,342   939,342          0.0  PushPop  :copy-in
     28.8          426,289          1  426,289.0  426,289.0   426,289   426,289          0.0  PushPop  :copy-out
      7.8          115,256          1  115,256.0  115,256.0   115,256   115,256          0.0  PushPop  :compute
```

This is the time the *CPU* spent inside each range. Copying in took 0.94 ms, copying out 0.43 ms, and "compute" 0.12 ms. So the copies win. But 0.12 ms for adding one million numbers on a GPU that moves about 1 TB/s looks too slow. Let's see what the GPU itself was doing:

```bash
nsys stats -q --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum vadd.nsys-rep
```

```text
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                       Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------
    100.0            4,160          1   4,160.0   4,160.0     4,160     4,160          0.0  add(const float *, const float *, float *, int)

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     69.1          756,390      2  378,195.0  378,195.0   374,915   381,475      4,638.6  [CUDA memcpy Host-to-Device]
     30.9          338,818      1  338,818.0  338,818.0   338,818   338,818          0.0  [CUDA memcpy Device-to-Host]
```

The kernel ran for **4.2 µs**. The three copies took **1.1 ms** on the GPU. The kernel is less than 0.4% of the GPU's busy time. The other 111 µs of our `compute` range were spent on the CPU side, mostly before the kernel started. We will find out where in the next section.

There is one more table worth a look, the CUDA API summary. It shows how long each CUDA *call* took on the CPU:

```bash
nsys stats -q --report cuda_api_sum vadd.nsys-rep
```

```text
 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)   Min (ns)   Max (ns)    StdDev (ns)            Name
 --------  ---------------  ---------  ------------  ---------  --------  -----------  ------------  ----------------------
     98.6      121,288,629          3  40,429,543.0  114,685.0   104,155  121,069,789  69,836,501.8  cudaMalloc
      1.1        1,351,314          3     450,438.0  453,841.0   425,588      471,885      23,335.3  cudaMemcpy
      0.2          300,253          3     100,084.3  109,365.0    80,010      110,878      17,401.3  cudaFree
      0.0           53,470          1      53,470.0   53,470.0    53,470       53,470           0.0  cudaLaunchKernel
      0.0           33,974          1      33,974.0   33,974.0    33,974       33,974           0.0  cuLibraryLoadData
      0.0            3,647          1       3,647.0    3,647.0     3,647        3,647           0.0  cudaDeviceSynchronize
```

The slowest thing in the whole program is `cudaMalloc`: 121 ms. But look at the median: 115 µs. Only the *first* call was slow. The first CUDA call in a program creates the CUDA *context*, which sets up the driver and the GPU for our process. Every program pays this once, and whichever CUDA call comes first gets the bill. This is also why we will always do a warm-up run before timing anything, as we did in post 03.

If you only want these tables, `nsys profile --stats=true ./vector-add-nvtx` records the profile and prints the default set of tables in one step. It is the closest thing to the old `nvprof ./a.out`.

---

## 4. Reading the Timeline

The summaries add up time, but they hide *order* and *gaps*. The `cuda_gpu_trace` report lists every GPU operation with its start time:

```bash
nsys stats -q --report cuda_gpu_trace vadd.nsys-rep
```

The real output has many columns (grid size, registers, device, stream, ...). Trimmed to the useful ones:

```text
 Start (ns)   Duration (ns)  GrdX   BlkX  Bytes (MB)  Throughput (MB/s)  SrcMemKd  DstMemKd  Name
 -----------  -------------  -----  ----  ----------  -----------------  --------  --------  -----------------------------------------------
 431,239,999        381,475                    4.000         10,484.000  Pageable  Device    [CUDA memcpy Host-to-Device]
 431,725,635        374,915                    4.000         10,668.000  Pageable  Device    [CUDA memcpy Host-to-Device]
 432,174,631          4,160  3,907   256                                                     add(const float *, const float *, float *, int)
 432,189,959        338,818                    4.000         11,804.000  Device    Pageable  [CUDA memcpy Device-to-Host]
```

We can read three things from this.

**The copies are slow.** Each 4 MB copy runs at about 10 to 12 GB/s. That is PCIe speed, roughly 100 times slower than the GPU's own memory. The `Pageable` column tells us why it is not even faster: our `std::vector` lives in normal (pageable) memory, so the driver first copies it into a special pinned buffer and then sends that over PCIe. We will fix this with pinned memory in post 12.

**There are gaps.** Subtract each operation's end from the next one's start:

| Between                         | Gap     | Why                                                        |
| ------------------------------- | ------- | ---------------------------------------------------------- |
| first copy and second copy      | 104 µs  | CPU-side work of the second `cudaMemcpy` of pageable memory |
| second copy and the kernel      | 74 µs   | first launch: `cuLibraryLoadData` and `cudaLaunchKernel`    |
| kernel and the copy back        | 11 µs   | `cudaDeviceSynchronize`, NVTX, and the next `cudaMemcpy`    |

During these gaps the GPU does nothing. The 74 µs one explains our slow `compute` range. Since CUDA 12.2, kernels are loaded *lazily*: the code of `add` is only loaded on the GPU the first time we launch it. That is the `cuLibraryLoadData` call (34 µs) in the API table, plus a slow first `cudaLaunchKernel` (53 µs). A second launch would not pay it.

**The kernel is suspiciously fast.** It reads two 4 MB arrays and writes one: 12 MB in 4.16 µs. That is about 2.9 TB/s, almost three times the ~1 TB/s of the 4090's memory. The data did not come from the GPU memory at all. The RTX 4090 has a 72 MB L2 cache, and we had just copied the inputs in, so they were still sitting in L2. Keep this in mind whenever a tiny benchmark looks too good: the data may be in the cache. Nsight Compute avoids this trap by flushing the caches before it measures, as we will see next.

---

## 5. One Kernel in Depth with Nsight Compute

`nsys` told us the kernel takes 4 µs. `ncu` tells us why. It runs the kernel several times, each time reading a different group of hardware counters, and then reports hundreds of metrics.

Profile only the `add` kernel, and only its first launch:

```bash
ncu -k add -c 1 ./vector-add-nvtx
```

* `-k add` picks kernels by name. Real programs launch many kernels; you rarely want all of them. `-k regex:add` matches by regular expression instead.
* `-c 1` stops after one profiled launch. Without it, a loop that launches a kernel 1000 times gets profiled 1000 times, and each one is slow.
* `-s N` skips the first N matching launches. This is useful to skip the cold first launch.

By default `ncu` collects the `basic` set of sections. `--set full` collects everything:

```bash
ncu --set full -k add -c 1 -o add-report ./vector-add-nvtx
```

`-o add-report` saves the result to `add-report.ncu-rep` instead of printing it. You can print it later with `ncu --import add-report.ncu-rep`, or open it in the GUI. To see which sections each set contains:

```bash
ncu --list-sets
```

```text
---------- --------------------------------------------------------------------------- ------- -----------------
Identifier Sections                                                                    Enabled Estimated Metrics
---------- --------------------------------------------------------------------------- ------- -----------------
basic      LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution                  yes     213
detailed   ComputeWorkloadAnalysis, LaunchStats, MemoryWorkloadAnalysis, MemoryWorkloa no      996
           dAnalysis_Chart, Occupancy, SourceCounters, SpeedOfLight, SpeedOfLight_Roof
           lineChart, Tile, WorkloadDistribution
full       ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnaly no      8051
           sis, MemoryWorkloadAnalysis_Chart, MemoryWorkloadAnalysis_Tables, NumaAffin
           ity, Nvlink_Tables, Nvlink_Topology, Occupancy, PmSampling, SchedulerStats,
            SourceCounters, SpeedOfLight, SpeedOfLight_HierarchicalDoubleRooflineChart
           ...
```

`full` collects about 8000 metrics instead of 213, so it replays the kernel many more times. For a 4 µs kernel that does not matter. For a long one, start with the default set.

### The permission error

The first time you run `ncu` on a desktop Linux machine, you will most likely see this:

```text
==PROF== Connected to process 343479 (/home/user/vector-add/vector-add-nvtx)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
Launched 3907 blocks x 256 threads for 1000000 elements
PASSED (0 errors)
==PROF== Disconnected from process 343479
```

The program still runs, but nothing is profiled. By default, the NVIDIA driver only lets root read the GPU performance counters. You can check it:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

```text
RmProfilingAdminOnly: 1
```

There are two fixes:

1. **Run as root, once.** `sudo` resets `PATH`, so give the full path: `sudo /usr/local/cuda/bin/ncu -k add -c 1 ./vector-add-nvtx`.
2. **Allow normal users, permanently.** Set a driver option, rebuild the initramfs, and reboot:

   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo update-initramfs -u
   sudo reboot
   ```

   After the reboot, `RmProfilingAdminOnly` shows `0` and `ncu` works without `sudo`.

`nsys` does not need this. It only records when things happened, not the hardware counters.

### What to look at

The report is long. For a first look, three sections answer most questions.

**GPU Speed Of Light Throughput.** "Speed of light" means the hardware peak. This section shows the kernel's `Duration`, plus `Memory Throughput` and `Compute (SM) Throughput`, each as a percentage of the peak. The larger of the two tells you what limits the kernel. Vector add does one addition for every 12 bytes it moves, so we expect `Memory Throughput` to be the high one and `Compute (SM) Throughput` to be low: the kernel is *memory-bound*. The `DRAM Throughput` line shows how much of that is the actual GPU memory.

Note that `ncu` measures under different conditions than `nsys`. By default it flushes all caches before each replay (`--cache-control all`) and locks the clocks to the base frequency (`--clock-control base`). So its `Duration` will not match the 4.16 µs from `nsys`. That is on purpose: flushing the cache means the inputs are *not* waiting in L2, so we measure the real memory traffic, and fixed clocks make runs comparable with each other.

**Occupancy.** Each SM can hold a limited number of warps at once. `Theoretical Occupancy` is how many it *could* hold with our block size and register use, and `Achieved Occupancy` is how many it actually held on average. The `Block Limit Registers`, `Block Limit Shared Mem`, and `Block Limit Warps` lines tell you which resource is the limit. Our kernel uses only 16 registers per thread (the `Reg/Trd` column in the `nsys` trace) and no shared memory, so nothing should limit it. We will study occupancy properly in post 08.

**Warp State Statistics** (in `--set full`). When a warp cannot issue its next instruction, it is *stalled*, and `ncu` records why. For a memory-bound kernel like ours, the top reason should be `Stall Long Scoreboard`, which means the warp is waiting for a load from global memory. That is the kernel spending its time waiting for data, which matches what Speed Of Light told us.

`ncu` also adds short hints under each section. They are worth reading, but check them against the numbers; they are rules of thumb, not proof.

---

## 6. The GUIs

Both tools have a graphical version that opens the same report files:

```bash
nsys-ui vadd.nsys-rep
ncu-ui add-report.ncu-rep
```

`nsys-ui` draws the timeline we read from text in section 4: one row for CUDA API calls, one for the GPU with our copies and kernel, and one for our NVTX ranges. You can zoom and see the gaps directly. It is the best way to explore a program you do not know yet. `ncu-ui` shows the same sections as the command line, with charts, and can put two reports side by side to compare a kernel before and after a change.

A common workflow is to profile on the machine with the GPU using the command line, then copy the report files to your laptop and open them there.

---

## 7. Exercise

Profile the original program from post 01, which has no NVTX ranges, and find how its time splits between copying and computing:

```bash
nsys profile -t cuda -o p01 ./vector-add
nsys stats -q --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum p01.nsys-rep
```

Before you run it, make a guess. Then add up the three copies and compare them with the kernel.

On my machine:

```text
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                       Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------
    100.0            7,265          1   7,265.0   7,265.0     7,265     7,265          0.0  add(const float *, const float *, float *, int)

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     68.9          819,399      2  409,699.5  409,699.5   409,667   409,732         46.0  [CUDA memcpy Host-to-Device]
     31.1          370,467      1  370,467.0  370,467.0   370,467   370,467          0.0  [CUDA memcpy Device-to-Host]
```

The copies take 1.19 ms and the kernel 7.3 µs, so about 99.4% of the GPU time is copying. The kernel time is different from our NVTX run (7.3 µs instead of 4.2 µs); a single cold launch of a tiny kernel is noisy, which is exactly why post 03 averaged many runs.

The lesson is uncomfortable but important: the GPU spent over 150 times longer moving the data than adding it. For a kernel this simple, the copies decide the speed of the whole program, and making the kernel faster would change almost nothing. A GPU pays off when the data stays on the device for many kernels, or when each byte needs a lot of work. Next, try the same with `-t cuda,osrt` and look at the `cudaMalloc` line in `cuda_api_sum`: how much of the whole program's run time is context creation?

In summary, profile before you optimize. Start with `nsys` to see the whole program: which operations take the time, and where the GPU sits idle. NVTX ranges put your own names on that timeline for the cost of one header. Then point `ncu` at the one kernel that matters, with `-k` and `-c`, and read Speed Of Light, Occupancy, and the warp stall reasons to learn *why* it is slow. Old tutorials that use `nvprof` or the Visual Profiler are out of date: both tools were removed in CUDA 13.0. In the next post, we will use `ncu` to see how the way a warp accesses memory decides a kernel's speed.

[1]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "Nsight Systems User Guide"
[2]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html "Nsight Compute CLI"
[3]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "Nsight Compute Profiling Guide"
[4]: https://nvidia.github.io/NVTX/ "NVTX documentation"
[5]: https://developer.nvidia.com/ERR_NVGPUCTRPERM "Permission issue with Performance Counters"
[6]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html "CUDA Toolkit Release Notes"
