#include <cstdio>
#include <memory>
#include <vector>
// #region shim
// Workaround for -U_GNU_SOURCE (see post 00): CUB pulls in <mutex>, which
// calls two glibc functions that are only declared when _GNU_SOURCE is set.
#include <pthread.h>
extern "C" int pthread_cond_clockwait(pthread_cond_t*, pthread_mutex_t*,
                                      clockid_t, const timespec*);
extern "C" int pthread_mutex_clocklock(pthread_mutex_t*, clockid_t,
                                       const timespec*);
// #endregion
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

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

constexpr int kThreads = 256;

// #region atomic
__global__ void reduce_atomic(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, in[i]);
    }
}
// #endregion

// #region shared
__global__ void reduce_shared(const int* in, int* out, int n) {
    __shared__ int s[kThreads];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, s[0]);
    }
}
// #endregion

// #region shuffle
__device__ int warp_sum(int v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;  // only lane 0 holds the full sum
}

__global__ void reduce_shuffle(const int* in, int* out, int n) {
    __shared__ int warp_sums[kThreads / 32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int v = warp_sum((i < n) ? in[i] : 0);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        v = warp_sum((lane < kThreads / 32) ? warp_sums[lane] : 0);
        if (lane == 0) {
            atomicAdd(out, v);
        }
    }
}
// #endregion

// #region cg
__global__ void reduce_cg(const int* in, int* out, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int v = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        v += in[i];
    }

    v = cg::reduce(warp, v, cg::plus<int>());
    if (warp.thread_rank() == 0) {
        atomicAdd(out, v);
    }
}
// #endregion

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

int main() {
    // #region host
    constexpr int n = 1 << 26;
    constexpr std::size_t bytes = n * sizeof(int);

    std::vector<int> h_in(n);
    long long expected = 0;
    for (int i = 0; i < n; ++i) {
        h_in[i] = i % 4;
        expected += h_in[i];
    }
    // #endregion

    auto d_in = make_device_buffer<int>(n);
    auto d_copy = make_device_buffer<int>(n);
    auto d_out = make_device_buffer<int>(1);
    if (!d_in || !d_copy || !d_out) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), bytes, cudaMemcpyHostToDevice));

    int mem_clock_khz, bus_bits;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    std::printf("n = %d ints (%zu MB), expected sum = %lld\n\n",
                n, bytes / 1'000'000, expected);

    // #region ceiling
    float copy_ms = time_ms([&] {
        cudaMemcpy(d_copy.get(), d_in.get(), bytes, cudaMemcpyDeviceToDevice);
    });
    double copy_gbs = 2.0 * bytes / (copy_ms * 1e-3) / 1e9;  // read + write
    std::printf("%-24s %.3f ms, %4.0f GB/s (%.1f%% of peak)\n", "copy (ceiling):",
                copy_ms, copy_gbs, 100.0 * copy_gbs / peak_gbs);
    // #endregion

    // #region report
    int failures = 0;
    auto report = [&](const char* name, float ms) {
        int sum = 0;
        cudaMemcpy(&sum, d_out.get(), sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = (sum == expected);
        failures += !ok;
        double gbs = bytes / (ms * 1e-3) / 1e9;  // every input byte read once
        std::printf("%-24s %.3f ms, %4.0f GB/s (%.1f%% of peak), sum = %d %s\n",
                    name, ms, gbs, 100.0 * gbs / peak_gbs, sum, ok ? "OK" : "WRONG");
    };
    // #endregion

    // #region run
    int* in = d_in.get();
    int* out = d_out.get();
    constexpr int blocks = (n + kThreads - 1) / kThreads;

    report("v1 global atomics:", time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(int));
        reduce_atomic<<<blocks, kThreads>>>(in, out, n);
    }));
    report("v2 shared tree:", time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(int));
        reduce_shared<<<blocks, kThreads>>>(in, out, n);
    }));
    report("v3 warp shuffle:", time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(int));
        reduce_shuffle<<<blocks, kThreads>>>(in, out, n);
    }));
    // #endregion

    // #region cg-launch
    int sms = 0, per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, reduce_cg, kThreads, 0));
    int cg_blocks = sms * per_sm;

    report("v4 cooperative groups:", time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(int));
        reduce_cg<<<cg_blocks, kThreads>>>(in, out, n);
    }));
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    // #region cub
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, in, out, n));
    auto d_temp = make_device_buffer<char>(temp_bytes);
    if (!d_temp) {
        return 1;
    }

    report("cub::DeviceReduce::Sum:", time_ms([&] {
        cub::DeviceReduce::Sum(d_temp.get(), temp_bytes, in, out, n);
    }));
    // #endregion
    CUDA_CHECK(cudaGetLastError());

    std::printf("\nv4 grid: %d SMs x %d blocks = %d blocks\n", sms, per_sm, cg_blocks);
    std::printf("CUB temp storage: %zu bytes\n", temp_bytes);
    std::printf("%s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
