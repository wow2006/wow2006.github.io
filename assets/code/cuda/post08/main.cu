#include <cstdio>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err));                         \
            return 1;                                                      \
        }                                                                  \
    } while (0)

using DeviceBuffer = std::unique_ptr<float, decltype(&cudaFree)>;

DeviceBuffer make_device_buffer(std::size_t count) {
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
    }
    return DeviceBuffer{ptr, &cudaFree};
}

// Average GPU time of one call to `work`, in milliseconds (post 03).
template <typename F>
float time_ms(F work, int warmup = 3, int reps = 20) {
    for (int i = 0; i < warmup; ++i) {
        work();
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) {
        work();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

// #region kernel
// y = a * x. Each thread handles ILP elements, spaced blockDim.x apart so
// every warp load stays coalesced. All ILP loads are issued before the
// first store, so they are in flight at the same time.
template <int ILP, int MinBlocks = 1>
__global__ void __launch_bounds__(256, MinBlocks)
scale(const float* x, float* y, float a, int n) {
    int base = blockIdx.x * blockDim.x * ILP + threadIdx.x;
    float v[ILP];
#pragma unroll
    for (int k = 0; k < ILP; ++k) {
        int i = base + k * blockDim.x;
        v[k] = i < n ? x[i] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < ILP; ++k) {
        int i = base + k * blockDim.x;
        if (i < n) {
            y[i] = a * v[k];
        }
    }
}
// #endregion

constexpr int threads = 256;

// #region run
// Launches scale<ILP, MinBlocks> with `smem` bytes of dynamic shared memory
// per block (the kernel never touches it), prints the occupancy the runtime
// predicts and the bandwidth we measure. Returns false on a CUDA error.
template <int ILP, int MinBlocks = 1>
bool run(const float* x, float* y, int n, int smem, const cudaDeviceProp& p,
         double peak_gbs) {
    auto kernel = scale<ILP, MinBlocks>;
    cudaFuncAttributes attr{};
    int blocksPerSM = 0;
    if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess ||
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel,
                                                      threads, smem) != cudaSuccess) {
        return false;
    }
    int warps = blocksPerSM * threads / p.warpSize;
    int maxWarps = p.maxThreadsPerMultiProcessor / p.warpSize;

    int blocks = (n + threads * ILP - 1) / (threads * ILP);
    float ms = time_ms([&] { kernel<<<blocks, threads, smem>>>(x, y, 2.0f, n); });
    double gbs = 2.0 * n * sizeof(float) / (ms * 1e6);  // read x + write y

    std::printf("scale<%2d,%d> regs=%2d smem=%6d  %d blocks/SM  %2d/%d warps "
                "(%5.1f%%)  %.3f ms, %3.0f GB/s (%4.1f%% of peak)\n",
                ILP, MinBlocks, attr.numRegs, smem, blocksPerSM, warps, maxWarps,
                100.0 * warps / maxWarps, ms, gbs, 100.0 * gbs / peak_gbs);
    return cudaGetLastError() == cudaSuccess;
}
// #endregion

int main() {
    // #region props
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("%s (sm_%d%d), %d SMs\n", p.name, p.major, p.minor,
                p.multiProcessorCount);
    std::printf("  max threads per SM      : %d (%d warps)\n",
                p.maxThreadsPerMultiProcessor,
                p.maxThreadsPerMultiProcessor / p.warpSize);
    std::printf("  max blocks per SM       : %d\n", p.maxBlocksPerMultiProcessor);
    std::printf("  registers per SM        : %d\n", p.regsPerMultiprocessor);
    std::printf("  shared memory per SM    : %zu bytes\n",
                p.sharedMemPerMultiprocessor);
    std::printf("  shared memory per block : %zu bytes (opt-in max)\n",
                p.sharedMemPerBlockOptin);
    std::printf("  reserved smem per block : %zu bytes\n\n",
                p.reservedSharedMemPerBlock);
    // #endregion

    // Peak bandwidth, as in post 03.
    int mem_clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    // #region potential
    int minGrid = 0, bestBlock = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock,
                                                  scale<1>, 0, 0));
    std::printf("MaxPotentialBlockSize(scale<1>) : block=%d, min grid=%d\n",
                bestBlock, minGrid);
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock,
                                                  scale<32>, 0, 0));
    std::printf("MaxPotentialBlockSize(scale<32>): block=%d, min grid=%d\n\n",
                bestBlock, minGrid);
    // #endregion

    constexpr int n = 1 << 26;  // 64M floats = 256 MB per buffer
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i % 1000);
    }
    auto d_x = make_device_buffer(n);
    auto d_y = make_device_buffer(n);
    if (!d_x || !d_y) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_x.get(), h.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // #region sweep
    // Allow 6, 5, ..., 1 resident blocks per SM by giving each block a
    // slice of the SM's shared memory. 6 x 256 threads already fill an SM.
    for (int want = 6; want >= 1; --want) {
        int smem = 0;
        if (want < 6) {
            smem = static_cast<int>(p.sharedMemPerMultiprocessor / want -
                                    p.reservedSharedMemPerBlock);
            smem = smem / 1024 * 1024;  // round down to whole KB
        }
        if (!run<1>(d_x.get(), d_y.get(), n, smem, p, peak_gbs) ||
            !run<4>(d_x.get(), d_y.get(), n, smem, p, peak_gbs)) {
            std::fprintf(stderr, "CUDA error\n");
            return 1;
        }
    }
    std::printf("\n");
    // #endregion

    // #region registers
    // 32 loads in flight per thread need many registers, so registers now
    // limit occupancy. MinBlocks = 6 forces the compiler to use fewer.
    if (!run<32>(d_x.get(), d_y.get(), n, 0, p, peak_gbs) ||
        !run<32, 6>(d_x.get(), d_y.get(), n, 0, p, peak_gbs)) {
        std::fprintf(stderr, "CUDA error\n");
        return 1;
    }
    // #endregion

    // #region verify
    CUDA_CHECK(cudaMemcpy(h.data(), d_y.get(), n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] != 2.0f * static_cast<float>(i % 1000)) {
            ++errors;
        }
    }
    std::printf("%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    // #endregion
    return errors == 0 ? 0 : 1;
}
