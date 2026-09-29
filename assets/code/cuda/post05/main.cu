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

// Average GPU time of one call to `work`, in milliseconds (from post 03).
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

constexpr double peak_gbs = 1008.0;  // RTX 4090, computed in post 03

// #region strided
__global__ void strided_copy(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[static_cast<long long>(i) * stride];
    }
}
// #endregion

// #region copy2d
constexpr int TILE = 32;  // each block covers a 32 x 32 tile of the matrix
constexpr int ROWS = 8;   // with 32 x 8 threads, so each thread does 4 rows

// Baseline: same access pattern as a transpose, but no transpose.
__global__ void copy_2d(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[(y + j) * n + x] = in[(y + j) * n + x];
    }
}

// #endregion

// #region transpose
// Neighbouring threads read neighbouring addresses; writes jump by n.
__global__ void transpose_read_coalesced(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[x * n + (y + j)] = in[(y + j) * n + x];
    }
}

// Neighbouring threads write neighbouring addresses; reads jump by n.
__global__ void transpose_write_coalesced(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[(y + j) * n + x] = in[x * n + (y + j)];
    }
}
// #endregion

int main() {
    // #region strided-main
    // 32M floats (128 MB) copied per launch, well above the 72 MB L2 cache.
    // The input must be 32x bigger (4 GB) for stride 32.
    constexpr int n = 1 << 25;
    constexpr int max_stride = 32;
    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;

    auto d_in = make_device_buffer(static_cast<std::size_t>(n) * max_stride);
    auto d_out = make_device_buffer(n);
    if (!d_in || !d_out) {
        return 1;
    }
    CUDA_CHECK(cudaMemset(d_in.get(), 0, sizeof(float) * n * max_stride));

    for (int stride = 1; stride <= max_stride; stride *= 2) {
        float ms = time_ms([&] {
            strided_copy<<<blocks, threads>>>(d_in.get(), d_out.get(), n, stride);
        });
        CUDA_CHECK(cudaGetLastError());
        // Useful bytes: n floats read + n floats written.
        double gbs = 2.0 * n * sizeof(float) / (ms * 1e6);
        std::printf("stride %2d: %.3f ms, %4.0f GB/s (%4.1f%% of peak)\n",
                    stride, ms, gbs, 100.0 * gbs / peak_gbs);
    }
    // #endregion

    // #region transpose-setup
    constexpr int size = 8192;  // 8192 x 8192 floats = 256 MB per matrix
    constexpr std::size_t count = static_cast<std::size_t>(size) * size;
    std::vector<float> h_in(count), h_out(count);
    for (std::size_t i = 0; i < count; ++i) {
        h_in[i] = static_cast<float>(i);
    }
    auto d_a = make_device_buffer(count);
    auto d_b = make_device_buffer(count);
    if (!d_a || !d_b) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_in.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));

    // #endregion

    // #region transpose-run
    dim3 grid(size / TILE, size / TILE);
    dim3 block(TILE, ROWS);
    struct Kernel {
        const char* name;
        void (*fn)(const float*, float*, int);
        bool transposes;
    };
    Kernel kernels[] = {
        {"copy_2d", copy_2d, false},
        {"transpose_read_coalesced", transpose_read_coalesced, true},
        {"transpose_write_coalesced", transpose_write_coalesced, true},
    };

    std::printf("\n");
    for (const Kernel& k : kernels) {
        float ms = time_ms([&] {
            k.fn<<<grid, block>>>(d_a.get(), d_b.get(), size);
        });
        CUDA_CHECK(cudaGetLastError());
        double gbs = 2.0 * count * sizeof(float) / (ms * 1e6);

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_b.get(), count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                float expected = k.transposes ? h_in[c * size + r] : h_in[r * size + c];
                if (h_out[r * size + c] != expected) {
                    ++errors;
                }
            }
        }
        std::printf("%s: %.3f ms, %.0f GB/s (%.1f%% of peak) %s\n", k.name, ms,
                    gbs, 100.0 * gbs / peak_gbs, errors == 0 ? "PASSED" : "FAILED");
    }
    // #endregion
    return 0;
}
