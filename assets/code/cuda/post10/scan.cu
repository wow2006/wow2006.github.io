#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

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

constexpr int BLOCK = 1024;  // elements scanned per block, one per thread

// #region hillis-steele
__global__ void scan_block_shared(const int* in, int* out, int* block_sums, int n) {
    __shared__ int s[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    s[threadIdx.x] = i < n ? in[i] : 0;
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset *= 2) {
        int left = threadIdx.x >= offset ? s[threadIdx.x - offset] : 0;
        __syncthreads();  // everyone has read before anyone writes
        s[threadIdx.x] += left;
        __syncthreads();  // everyone has written before the next read
    }

    if (i < n) {
        out[i] = s[threadIdx.x];
    }
    if (block_sums && threadIdx.x == BLOCK - 1) {
        block_sums[blockIdx.x] = s[BLOCK - 1];
    }
}
// #endregion

// #region warp-scan
__device__ int warp_scan(int v) {
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        int left = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) {
            v += left;
        }
    }
    return v;
}

__global__ void scan_block_warp(const int* in, int* out, int* block_sums, int n) {
    __shared__ int warp_sums[BLOCK / 32];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int v = warp_scan(i < n ? in[i] : 0);        // 1. scan inside each warp
    if (lane == 31) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        warp_sums[lane] = warp_scan(warp_sums[lane]);  // 2. scan the 32 warp totals
    }
    __syncthreads();
    if (warp > 0) {
        v += warp_sums[warp - 1];                // 3. add the warps to our left
    }

    if (i < n) {
        out[i] = v;
    }
    if (block_sums && threadIdx.x == BLOCK - 1) {
        block_sums[blockIdx.x] = v;
    }
}
// #endregion

// #region multi-block
__global__ void add_offsets(int* out, const int* block_sums, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        out[i] += block_sums[blockIdx.x - 1];
    }
}

using ScanKernel = void (*)(const int*, int*, int*, int);

// Inclusive scan of n ints. scratch holds the block sums of every level.
void scan(ScanKernel kernel, const int* in, int* out, int n, int* scratch) {
    int blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks == 1) {
        kernel<<<1, BLOCK>>>(in, out, nullptr, n);
        return;
    }
    int* block_sums = scratch;
    kernel<<<blocks, BLOCK>>>(in, out, block_sums, n);        // 1. scan each block
    scan(kernel, block_sums, block_sums, blocks, scratch + blocks);  // 2. scan the block sums
    add_offsets<<<blocks, BLOCK>>>(out, block_sums, n);       // 3. add them back
}
// #endregion

int main() {
    // #region host
    constexpr int n = 30'000'000;
    constexpr std::size_t bytes = n * sizeof(int);

    std::vector<int> in(n), out(n), expected(n);
    std::mt19937 rng(42);
    for (int& x : in) {
        x = rng() % 10;
    }
    std::inclusive_scan(in.begin(), in.end(), expected.begin());
    // #endregion

    // #region scratch
    int scratch_n = 0;
    for (int m = n; m > 1; m = (m + BLOCK - 1) / BLOCK) {
        scratch_n += (m + BLOCK - 1) / BLOCK;
    }
    // #endregion

    auto d_in = make_device_buffer<int>(n);
    auto d_out = make_device_buffer<int>(n);
    auto d_scratch = make_device_buffer<int>(scratch_n);
    if (!d_in || !d_out || !d_scratch) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in.get(), in.data(), bytes, cudaMemcpyHostToDevice));

    int mem_clock_khz, bus_bits;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    std::printf("n = %d ints (%zu MB), scratch = %d ints\n\n", n, bytes / 1'000'000, scratch_n);

    // #region report
    int failures = 0;
    auto report = [&](const char* name, float ms, bool check) {
        double gbs = 2.0 * bytes / (ms * 1e-3) / 1e9;  // read n + write n
        std::printf("%-24s %.3f ms, %4.0f GB/s (%.1f%% of peak)", name, ms, gbs,
                    100.0 * gbs / peak_gbs);
        if (check) {
            cudaMemcpy(out.data(), d_out.get(), bytes, cudaMemcpyDeviceToHost);
            bool ok = (out == expected);
            failures += !ok;
            std::printf(" %s", ok ? "OK" : "WRONG");
        }
        std::printf("\n");
    };
    // #endregion

    // #region run
    int* d = d_in.get();
    int* o = d_out.get();
    report("copy (ceiling):", time_ms([&] {
        cudaMemcpy(o, d, bytes, cudaMemcpyDeviceToDevice);
    }), false);
    report("v1 shared Hillis-Steele:", time_ms([&] {
        scan(scan_block_shared, d, o, n, d_scratch.get());
    }), true);
    report("v2 warp shuffle:", time_ms([&] {
        scan(scan_block_warp, d, o, n, d_scratch.get());
    }), true);
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    // #region cub
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d, o, n));
    auto d_temp = make_device_buffer<char>(temp_bytes);
    if (!d_temp) {
        return 1;
    }

    report("cub::DeviceScan:", time_ms([&] {
        cub::DeviceScan::InclusiveSum(d_temp.get(), temp_bytes, d, o, n);
    }), true);
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    std::printf("\nlast element = %d (expected %d)\n", out[n - 1], expected[n - 1]);
    std::printf("%s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
