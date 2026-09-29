#include <chrono>
#include <cstdio>
#include <memory>
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

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// #region time_ms
// Average GPU time of one call to `work`, in milliseconds.
// Errors are left for the caller to catch with cudaGetLastError().
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
// #endregion

int main() {
    // #region peak
    int sm_count, clock_khz, mem_clock_khz, bus_bits, major, minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));

    // Memory is double data rate: two transfers per clock.
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    // FP32 units per SM: 64 on Turing (7.5) and A100 (8.0), 128 on newer GPUs.
    int fp32_per_sm = (major == 7 || (major == 8 && minor == 0)) ? 64 : 128;
    // One fused multiply-add per unit per clock counts as 2 FLOPs.
    double peak_gflops = 2.0 * sm_count * fp32_per_sm * clock_khz * 1e3 / 1e9;

    std::printf("Peak memory bandwidth: %.0f GB/s\n", peak_gbs);
    std::printf("Peak FP32 compute:     %.0f GFLOP/s\n\n", peak_gflops);
    // #endregion

    constexpr int n = 1 << 25;  // 32M floats = 128 MB per buffer
    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;

    auto d_a = make_device_buffer(n);
    auto d_b = make_device_buffer(n);
    auto d_c = make_device_buffer(n);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }
    CUDA_CHECK(cudaMemset(d_a.get(), 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b.get(), 0, n * sizeof(float)));

    // #region work
    auto run_add = [&] {
        add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
    };
    // #endregion

    // #region cold
    float cold_ms = time_ms(run_add, 0, 1);
    CUDA_CHECK(cudaGetLastError());
    std::printf("First launch (cold):        %8.3f ms\n", cold_ms);
    // #endregion

    // #region chrono
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    run_add();
    auto t1 = clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clock::now();

    std::chrono::duration<double, std::milli> no_sync = t1 - t0;
    std::chrono::duration<double, std::milli> with_sync = t2 - t0;
    std::printf("std::chrono, no sync:       %8.3f ms\n", no_sync.count());
    std::printf("std::chrono, with sync:     %8.3f ms\n", with_sync.count());
    // #endregion

    // #region score
    float ms = time_ms(run_add);
    CUDA_CHECK(cudaGetLastError());
    std::printf("cudaEvent, 3 warm-up + 20:  %8.3f ms\n\n", ms);

    double bytes = 3.0 * n * sizeof(float);  // read a and b, write c
    double flops = 1.0 * n;                  // one add per element
    double gbs = bytes / (ms * 1e-3) / 1e9;
    double gflops = flops / (ms * 1e-3) / 1e9;
    std::printf("add: %.3f ms, %.0f GB/s (%.1f%% of peak), %.1f GFLOP/s (%.2f%% of peak)\n",
                ms, gbs, 100.0 * gbs / peak_gbs, gflops, 100.0 * gflops / peak_gflops);
    // #endregion

    // #region roofline
    double intensity = flops / bytes;  // FLOPs per byte moved
    double ridge = peak_gflops / peak_gbs;
    std::printf("\nArithmetic intensity of add: %.3f FLOP/byte\n", intensity);
    std::printf("Ridge point of this GPU:     %.1f FLOP/byte\n", ridge);
    std::printf("add is %s-bound\n", intensity < ridge ? "memory" : "compute");
    // #endregion
    return 0;
}
