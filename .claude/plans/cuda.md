# CUDA "Zero to Hero" Series — Plan (2026)

Location: `content/en/posts/cuda/` · code: `assets/code/cuda/postNN/{main.cu,CMakeLists.txt}`
Last updated: 2026-09-30

## 1. Current state

| # | File | Title | Covers |
|---|------|-------|--------|
| 00 | `post00-helloworld-cuda.md` | Helloworld CUDA in 2026 | Install CUDA 13.1 (Ubuntu 26.04, `cuda-toolkit` apt pkg), validate, `__global__`, `<<<blocks, threads>>>`, `blockIdx`/`threadIdx`, kernel `printf`, `CUDA_CHECK`, `cudaGetLastError`, `cudaDeviceSynchronize`, device query, compute capability, CMake (`CMAKE_CUDA_ARCHITECTURES=native`), `-U_GNU_SOURCE` workaround |
| 01 | `post01-moving-data-between-cpu-and-gpu.md` | Moving data between the CPU and the GPU | `cudaMalloc`/`cudaFree` via `std::unique_ptr` deleter, `cudaMemcpy` + directions, global index, grid rounding `(n+t-1)/t`, bounds check, vector add (1M floats), `compute-sanitizer` |

Audience: knows C++ (`std::vector`, `std::unique_ptr`) and basic Linux; zero CUDA.
Reference machine: RTX 4090 (sm_89, Ada, ~1 TB/s memory bandwidth). CUDA 13 minimum = compute capability 7.5 (Turing).

### Housekeeping
- [x] `weight: N+1` on post N (Hugo treats `weight: 0` as unset, so post00 = 1 … post21 = 22).
- [x] Front-matter consistent: `series: ["cuda"]`, `tags: [cuda, cpp, linux]`. All posts are `draft: true`.
- [x] Every post: "What We Will Learn", full program, build & run, exercise (same template as 00/01).

### Status (2026-09-30): posts 02–21 drafted
All 22 posts exist in `content/en/posts/cuda/` with code in `assets/code/cuda/postNN/`; `hugo --buildDrafts` builds clean. Outstanding:
- [ ] `ncu` numbers never captured (ERR_NVGPUCTRPERM). Posts 04, 05, 06, 08, 18 show commands + labelled *expected* values. Fix: `NVreg_RestrictProfilingToAdminUsers=0` or `sudo ncu`, then paste real output.
- [ ] CUB/Thrust under `-U_GNU_SOURCE` need a pthread declaration shim (post 09: 2 decls, post 11/17: 4 decls). Drop when the toolkit fixes the header clash.
- [ ] CMake without a build type compiles host code unoptimized (only matters for CPU timings; post 11/16 use `-DCMAKE_BUILD_TYPE=Release`).
- [ ] Post 17 WMMA = 44 TFLOP/s (col-major B) vs post 21 WMMA = ~71 (row-major B): unexplained.
- [ ] Post 20: TMA/cluster host code compiled for sm_90 but never run.
- [ ] Post 17 links the CUTLASS example on GitHub — works only once pushed to `main`.

## 2. Guiding rules for the series

1. **Measure everything.** From post 03 on, every kernel reports time + GB/s (or FLOP/s) vs. the 4090 peak. That number is the reader's "score".
2. **Profile early.** Nsight tools appear in post 04, not at the end. "Hero" = can explain *why* a kernel is slow.
3. **Timeless first, 2026 last.** Coalescing, shared memory, occupancy, patterns are the core. Newest tech (cuTile, TMA, tcgen05) goes at the end.
4. **Hand-write, then show the library.** After writing reduction/scan/GEMM by hand, show CUB/Thrust/cuBLAS doing it — and compare numbers.
5. **Runs on the reader's GPU.** Everything through post 19 must run on sm_75+ consumer cards. Hopper/Blackwell-only material is clearly marked "concepts" unless run on rented hardware.

## 3. The list

### Stage 1 — Basics
- **00 ✅ Hello world** — done.
- **01 ✅ Moving data** — done.
- **02 Threads in 2D & warps**
  - 2D grids/blocks (`dim3`), 2D indexing on an image (grayscale → box blur).
  - Grid-stride loops (one thread handles many elements; works for any size).
  - Warps = 32 threads executing together; branch divergence and why `if` inside a warp costs time.
  - Exercise: blur with different block shapes (8×8, 16×16, 32×8), compare.
- **03 Measuring speed properly**
  - `cudaEvent_t` timing, warm-up runs, averaging, why CPU timers lie with async launches.
  - Effective bandwidth (GB/s) and compute (GFLOP/s) vs. theoretical peak; a light roofline intro.
  - Exercise: bandwidth of vector add and copy kernel vs. `cudaMemcpy` device-to-device.
- **04 Profiling: Nsight Systems & Nsight Compute**
  - `nsys` = timeline (launches, copies, gaps). `ncu` = one kernel in depth (memory throughput, occupancy, stalls).
  - Note: `nvprof` and Visual Profiler were REMOVED in CUDA 13.0 — old tutorials using them are outdated.
  - Exercise: find the copy vs. compute time split of post 01.

### Stage 2 — Memory (where most speed is won or lost)
- **05 Global memory & coalescing**
  - How warps load memory; coalesced vs. strided access.
  - Naive matrix transpose: fast reads + slow writes (or the reverse). Measure with ncu.
- **06 Shared memory & `__syncthreads`**
  - Tiled transpose via shared memory; bank conflicts and the `+1` padding trick.
  - Show a real race (missing `__syncthreads`), catch it with `compute-sanitizer --tool racecheck`.
- **07 Tiled matrix multiply**
  - Naive → shared-memory tiled SGEMM. Compare with cuBLAS to show the gap (sets up post 18).
- **08 Occupancy & registers**
  - Occupancy = how many warps are resident per SM; limits from registers, shared memory, block size.
  - `__launch_bounds__`, `cudaOccupancyMaxPotentialBlockSize`, reading ncu occupancy section.
  - Lesson: higher occupancy is not always faster.

### Stage 3 — Common parallel patterns
- **09 Reduction (sum)**
  - Atomics → shared-memory tree → warp shuffle (`__shfl_down_sync`) → cooperative groups.
  - End: `cub::DeviceReduce` does it in one call; compare speeds.
- **10 Scan & histogram**
  - Prefix sum (block scan + multi-block), histogram with atomics and privatization (per-block copies, then merge).
- **11 CCCL & libraries: don't write what exists**
  - Thrust (STL-like algorithms), CUB (fast building blocks), libcu++ (CUDA std library: atomics, sync). CCCL 3.x is the current major version.
  - cuBLAS, cuFFT quick examples.

### Stage 4 — Doing work at the same time
- **12 Pinned memory, streams & async copies**
  - `cudaMallocHost`, `cudaMemcpyAsync`, multiple streams; overlap copy of batch N+1 with compute of batch N. Show it on the nsys timeline.
- **13 Unified memory**
  - `cudaMallocManaged`, page faults, `cudaMemPrefetchAsync`; when it is convenient and when it silently hurts.
- **14 CUDA Graphs**
  - Stream capture, instantiate, replay; cutting launch overhead for many small kernels.
- **15 (optional) Multi-GPU & P2P basics**
  - Note: nv-p2p APIs deprecated in CUDA 13.0, removal planned for 14.0 (kernel-driver-level API; check relevance before writing).

### Stage 5 — Hero level
- **16 Number formats**
  - FP32/FP16/BF16/FP8, `__half`, mixed precision, `--use_fast_math` trade-offs and accuracy checks.
- **17 Tensor Cores (on the 4090)**
  - WMMA API → `mma.sync` inline PTX → quick tour of CUTLASS.
- **18 Capstone: SGEMM from naive to near-cuBLAS**
  - One optimization per step (coalescing, shared tiles, register tiling, vectorized loads, double buffering / `cp.async`, …), each profiled with ncu and plotted vs. cuBLAS. The post people will bookmark.
- **19 (optional capstone) Fused softmax / mini FlashAttention**
- **20 Hopper & Blackwell concepts**
  - Thread block clusters, distributed shared memory (DSMEM), TMA (hardware bulk async copies), wgmma (Hopper), tcgen05 + tensor memory (data-center Blackwell).
  - These are reachable from the CuTe Python DSL (CUTLASS; public beta, expected to leave beta ~end of summer 2026 — verify).
  - Cannot run on a 4090 (sm_89). Mark as concepts, or run on rented H100/B200.
- **21 Tile programming in 2026: cuTile vs Triton vs CUDA C++**
  - cuTile shipped with CUDA 13.1 (Dec 2025). Reported as Python-only (C++ planned) and Blackwell-only (CC 10.x/12.x). Status/label not confirmed from an NVIDIA status page — re-verify before writing.
  - When to pick which: C++ for full control, Triton/cuTile for productivity on tile-shaped problems.

## 4. 2026 facts to keep in mind (verified by search, 2026-09-29)

- Latest toolkit: CUDA 13.x (docs show up to 13.4 Update 1). 13.1 (Dec 4, 2025) was a major release (CUDA Tile/cuTile).
- Removed in 13.0: `nvprof`, Visual Profiler; offline compile/library support for Maxwell/Pascal/Volta (use 12.x for those). Ubuntu 20.04 dropped.
- Deprecated: Nsight Eclipse Edition plugins (13.1); nv-p2p APIs (13.0, removal planned 14.0).
- Profilers today: Nsight Systems + Nsight Compute.
- CUDA Python: official `NVIDIA/cuda-python` (cuda.core, cuda.bindings) actively maintained; nvmath-python bindings still marked experimental.
- CCCL = Thrust + CUB + libcu++ in one repo; 3.x current.

## 5. Open questions / to verify before writing
- [x] cuTile: `cuda-tile` 1.6.0 (2026-09-09); CC 8.x added in 13.2, Hopper + CUDA Tile C++ in 13.3. Runs on the 4090 via pip `cuda-tile[tileiras]` (post 21).
- [x] sm_120/121 lack tcgen05/TMEM: ptxas rejects `tcgen05.alloc` for sm_120a/121a (post 20).
- [x] CuTe DSL still beta: `nvidia-cutlass-dsl` 4.8.0 (2026-09-21) "Development Status :: 4 - Beta" (post 20).
- [ ] Whether NVIDIA rewrote the CUDA Programming Guide (unconfirmed).

## 6. Reader resources to link
- *Programming Massively Parallel Processors*, 4th ed. (Hwu, Kirk, El Hajj, 2022) — Stages 2–3 follow it.
- GPU MODE lectures (community series on PMPP + modern kernels).
- NVIDIA CUDA Programming Guide & Best Practices Guide.

## 7. Sources
- CUDA release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA 13.0 notes: https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html
- CUDA 13.1 notes: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html
- cuTile: https://www.phoronix.com/news/NVIDIA-CUDA-13.1 · https://developer.nvidia.com/blog/tag/cuda-tile/
- CUDA Python: https://github.com/NVIDIA/cuda-python · https://nvidia.github.io/cuda-python/latest/
- nvmath-python bindings: https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html
- CCCL: https://github.com/nvidia/cccl · https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html
- CUTLASS / CuTe DSL: https://github.com/nvidia/cutlass · https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html · https://docs.nvidia.com/cutlass/4.6.2/CHANGELOG.html
- Colfax Blackwell clusters GEMM: https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
- GPU MODE notes: https://christianjmills.com/posts/cuda-mode-notes/lecture-002/
