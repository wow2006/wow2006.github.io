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

constexpr int TILE = 32;  // each block covers a 32 x 32 tile of the matrix
constexpr int ROWS = 8;   // with 32 x 8 threads, so each thread does 4 rows

// #region naive
// The ceiling: same access pattern as a transpose, but no transpose.
__global__ void copy_2d(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[(y + j) * n + x] = in[(y + j) * n + x];
    }
}

// From post 05: reads are coalesced, writes jump by n floats.
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[x * n + (y + j)] = in[(y + j) * n + x];
    }
}
// #endregion

// #region shared
template <int Pad>
__global__ void transpose_shared(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + Pad];

    // Load the tile: consecutive threads read consecutive addresses.
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    }

    __syncthreads();  // wait until all 256 threads have filled the tile

    // Store to the mirrored tile: consecutive threads write consecutive
    // addresses, reading a column of the tile to do it.
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
// #endregion

int main(int argc, char** argv) {
    int mem_clock_khz, bus_bits;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8) / 1e9;

    // #region setup
    // 8192 x 8192 floats = 256 MB per matrix, far bigger than the L2 cache.
    const int size = argc > 1 ? std::atoi(argv[1]) : 8192;
    if (size <= 0 || size % TILE != 0) {
        std::fprintf(stderr, "size must be a positive multiple of %d\n", TILE);
        return 1;
    }
    const std::size_t count = static_cast<std::size_t>(size) * size;
    const std::size_t bytes = count * sizeof(float);

    std::vector<float> h_in(count), h_out(count);
    for (std::size_t i = 0; i < count; ++i) {
        h_in[i] = static_cast<float>(i % 1'000'000);
    }
    auto d_in = make_device_buffer(count);
    auto d_out = make_device_buffer(count);
    if (!d_in || !d_out) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), bytes, cudaMemcpyHostToDevice));
    // #endregion

    // #region launch
    dim3 grid(size / TILE, size / TILE);  // one block per 32 x 32 tile
    dim3 block(TILE, ROWS);               // 32 x 8 = 256 threads
    struct Kernel {
        const char* name;
        void (*fn)(const float*, float*, int);
        bool transposes;
    };
    Kernel kernels[] = {
        {"copy_2d (ceiling)", copy_2d, false},
        {"transpose_naive", transpose_naive, true},
        {"transpose_shared<0>", transpose_shared<0>, true},
        {"transpose_shared<1>", transpose_shared<1>, true},
    };

    std::printf("Matrix %d x %d (%zu MB), peak %.0f GB/s\n", size, size,
                bytes >> 20, peak_gbs);
    std::printf("%-20s  time (ms)      GB/s   of peak\n", "kernel");
    int failures = 0;
    for (const Kernel& k : kernels) {
        CUDA_CHECK(cudaMemset(d_out.get(), 0, bytes));
        float ms = time_ms([&] {
            k.fn<<<grid, block>>>(d_in.get(), d_out.get(), size);
        });
        CUDA_CHECK(cudaGetLastError());
        // Every element is read once and written once.
        double gbs = 2.0 * bytes / (ms * 1e6);

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), bytes,
                              cudaMemcpyDeviceToHost));
        std::size_t errors = 0;
        for (std::size_t r = 0; r < std::size_t(size); ++r) {
            for (std::size_t c = 0; c < std::size_t(size); ++c) {
                float expected = k.transposes ? h_in[c * size + r] : h_in[r * size + c];
                errors += h_out[r * size + c] != expected;
            }
        }
        failures += errors != 0;
        std::printf("%-20s  %9.4f  %8.1f   %6.1f%%   %s", k.name, ms, gbs,
                    100.0 * gbs / peak_gbs, errors == 0 ? "PASSED" : "FAILED");
        if (errors != 0) {
            std::printf(" (%zu wrong elements)", errors);
        }
        std::printf("\n");
    }
    // #endregion
    return failures == 0 ? 0 : 1;
}
