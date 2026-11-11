#include <cstdio>
#include <cstdlib>
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

// #region buffer
using Buffer = std::unique_ptr<float, decltype(&cudaFree)>;

Buffer make_device_buffer(std::size_t count) {
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
    }
    return Buffer{ptr, &cudaFree};
}

Buffer make_managed_buffer(std::size_t count) {
    float* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, count * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
    }
    return Buffer{ptr, &cudaFree};
}
// #endregion

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void fill(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }
}

int count_errors(const float* c, int n) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f * i) {
            ++errors;
        }
    }
    return errors;
}

constexpr int n = 1 << 26;
constexpr std::size_t bytes = n * sizeof(float);
constexpr int threads = 256;
constexpr int blocks = (n + threads - 1) / threads;

constexpr double peak_gbs = 1008.0;  // RTX 4090, computed in post 03

// before_ms < 0 means "nothing was done before the kernel".
void report(const char* name, float before_ms, float kernel_ms, int errors) {
    double gbs = 3.0 * bytes / (kernel_ms * 1e-3) / 1e9;
    char before[32] = "-";
    if (before_ms >= 0) {
        std::snprintf(before, sizeof(before), "%.2f ms", before_ms);
    }
    std::printf("%-31s %10s %9.3f ms %6.0f GB/s (%5.1f%%)  %s\n", name, before,
                kernel_ms, gbs, 100.0 * gbs / peak_gbs,
                errors == 0 ? "PASSED" : "FAILED");
}

int main(int argc, char** argv) {
    // Run one case with `./unified-memory 2`, or all of them with no argument.
    auto run = [&](int k) { return argc < 2 || std::atoi(argv[1]) == k; };

    // #region query
    int device = 0;
    int concurrent = 0;
    int pageable = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &concurrent, cudaDevAttrConcurrentManagedAccess, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &pageable, cudaDevAttrPageableMemoryAccess, device));
    std::printf("concurrentManagedAccess = %d, pageableMemoryAccess = %d\n",
                concurrent, pageable);
    // #endregion
    std::printf("3 arrays x %zu MB, %d blocks x %d threads\n\n",
                bytes >> 20, blocks, threads);
    std::printf("%-31s %10s %12s %20s\n", "case", "before", "kernel",
                "bandwidth (% peak)");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto elapsed_ms = [&] {
        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    };

    // Warm-up: the first launch also loads the kernel onto the GPU.
    add<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // #region locations
    cudaMemLocation gpu{};
    gpu.type = cudaMemLocationTypeDevice;
    gpu.id = device;
    cudaMemLocation cpu{};
    cpu.type = cudaMemLocationTypeHost;
    // #endregion

    // 1. The post 01 way: cudaMalloc + cudaMemcpy.
    if (run(1)) {
        std::vector<float> a(n), b(n), c(n);
        fill(a.data(), b.data(), n);
        auto d_a = make_device_buffer(n);
        auto d_b = make_device_buffer(n);
        auto d_c = make_device_buffer(n);
        if (!d_a || !d_b || !d_c) {
            return 1;
        }

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        float copy_ms = elapsed_ms();

        CUDA_CHECK(cudaEventRecord(start));
        add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        float kernel_ms = elapsed_ms();

        CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
        report("cudaMalloc + cudaMemcpy", copy_ms, kernel_ms,
               count_errors(c.data(), n));
    }

    // 2. Managed memory, first touched by the CPU, no hints.
    if (run(2)) {
        // #region managed
        auto a = make_managed_buffer(n);
        auto b = make_managed_buffer(n);
        auto c = make_managed_buffer(n);
        if (!a || !b || !c) {
            return 1;
        }
        fill(a.get(), b.get(), n);  // the CPU writes straight into it

        CUDA_CHECK(cudaEventRecord(start));
        add<<<blocks, threads>>>(a.get(), b.get(), c.get(), n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        float first_ms = elapsed_ms();

        CUDA_CHECK(cudaEventRecord(start));
        add<<<blocks, threads>>>(a.get(), b.get(), c.get(), n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        float second_ms = elapsed_ms();

        CUDA_CHECK(cudaDeviceSynchronize());  // before the CPU reads c
        int errors = count_errors(c.get(), n);
        // #endregion
        report("managed, page faults", -1, first_ms, errors);
        report("managed, same data again", -1, second_ms, errors);
    }

    // 3. Managed memory, prefetched to the GPU before the kernel.
    if (run(3)) {
        auto a = make_managed_buffer(n);
        auto b = make_managed_buffer(n);
        auto c = make_managed_buffer(n);
        if (!a || !b || !c) {
            return 1;
        }
        fill(a.get(), b.get(), n);

        // #region prefetch
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemPrefetchAsync(a.get(), bytes, gpu, 0, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(b.get(), bytes, gpu, 0, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(c.get(), bytes, gpu, 0, 0));
        CUDA_CHECK(cudaEventRecord(stop));
        float prefetch_ms = elapsed_ms();
        // #endregion

        CUDA_CHECK(cudaEventRecord(start));
        add<<<blocks, threads>>>(a.get(), b.get(), c.get(), n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        float kernel_ms = elapsed_ms();

        CUDA_CHECK(cudaDeviceSynchronize());
        report("managed + cudaMemPrefetchAsync", prefetch_ms, kernel_ms,
               count_errors(c.get(), n));
    }

    // 4. Managed memory that stays on the CPU; the GPU reads it over PCIe.
    if (run(4)) {
        auto a = make_managed_buffer(n);
        auto b = make_managed_buffer(n);
        auto c = make_managed_buffer(n);
        if (!a || !b || !c) {
            return 1;
        }
        // #region advise
        for (float* p : {a.get(), b.get(), c.get()}) {
            CUDA_CHECK(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, cpu));
            CUDA_CHECK(cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, gpu));
        }
        // #endregion
        fill(a.get(), b.get(), n);

        CUDA_CHECK(cudaEventRecord(start));
        add<<<blocks, threads>>>(a.get(), b.get(), c.get(), n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        float kernel_ms = elapsed_ms();

        CUDA_CHECK(cudaDeviceSynchronize());
        report("managed, stays on CPU (advise)", -1, kernel_ms,
               count_errors(c.get(), n));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
