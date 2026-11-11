#include <cstdio>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cub/device/device_histogram.cuh>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err));                         \
            return 1;                                                      \
        }                                                                  \
    } while (0)

template <typename T>
using DeviceBuffer = std::unique_ptr<T, decltype(&cudaFree)>;

template <typename T>
DeviceBuffer<T> make_device_buffer(std::size_t count) {
    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
    }
    return DeviceBuffer<T>{ptr, &cudaFree};
}

// Average GPU time of one call to `work`, in milliseconds (from post 03).
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

constexpr int BINS = 256;

// #region global
__global__ void histogram_global(const unsigned char* data, unsigned int* bins, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&bins[data[i]], 1u);
    }
}
// #endregion

// #region shared
__global__ void histogram_shared(const unsigned char* data, unsigned int* bins, int n) {
    __shared__ unsigned int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        local[b] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        if (local[b] > 0) {
            atomicAdd(&bins[b], local[b]);
        }
    }
}
// #endregion

int main() {
    // #region host
    constexpr int n = 256 << 20;  // 256 MiB of bytes
    std::vector<unsigned char> data(n);
    std::mt19937 rng(42);
    for (auto& x : data) {
        x = rng() & 0xff;
    }
    std::vector<unsigned int> expected(BINS, 0), bins(BINS);
    for (unsigned char x : data) {
        ++expected[x];
    }
    // #endregion

    auto d_data = make_device_buffer<unsigned char>(n);
    auto d_bins = make_device_buffer<unsigned int>(BINS);
    if (!d_data || !d_bins) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_data.get(), data.data(), n, cudaMemcpyHostToDevice));

    int mem_clock_khz, bus_bits;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    std::printf("n = %d bytes (%d MB), %d bins\n\n", n, n / 1'000'000, BINS);

    // #region report
    int failures = 0;
    auto report = [&](const char* name, float ms) {
        cudaMemcpy(bins.data(), d_bins.get(), BINS * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        bool ok = (bins == expected);
        failures += !ok;
        double gbs = n / (ms * 1e-3) / 1e9;  // every input byte read once
        std::printf("%-24s %.3f ms, %4.0f GB/s (%.1f%% of peak) %s\n", name, ms, gbs,
                    100.0 * gbs / peak_gbs, ok ? "OK" : "WRONG");
    };
    // #endregion

    // #region run
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256;
    const int blocks = sms * 8;
    unsigned char* d = d_data.get();
    unsigned int* b = d_bins.get();

    report("v1 global atomics:", time_ms([&] {
        cudaMemsetAsync(b, 0, BINS * sizeof(unsigned int));
        histogram_global<<<blocks, threads>>>(d, b, n);
    }));
    report("v2 shared atomics:", time_ms([&] {
        cudaMemsetAsync(b, 0, BINS * sizeof(unsigned int));
        histogram_shared<<<blocks, threads>>>(d, b, n);
    }));
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    // #region cub
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(nullptr, temp_bytes, d, b,
                                                   BINS + 1, 0, BINS, n));
    auto d_temp = make_device_buffer<char>(temp_bytes);
    if (!d_temp) {
        return 1;
    }

    report("cub::DeviceHistogram:", time_ms([&] {
        cub::DeviceHistogram::HistogramEven(d_temp.get(), temp_bytes, d, b,
                                            BINS + 1, 0, BINS, n);
    }));
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    std::printf("\ngrid: %d blocks x %d threads\n", blocks, threads);
    std::printf("bins[0] = %u, bins[255] = %u\n", bins[0], bins[255]);
    std::printf("%s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
