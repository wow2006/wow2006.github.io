#include <cmath>
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

// #region pinned
using PinnedBuffer = std::unique_ptr<float, decltype(&cudaFreeHost)>;

PinnedBuffer make_pinned_buffer(std::size_t count) {
    float* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, count * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocHost: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
    }
    return PinnedBuffer{ptr, &cudaFreeHost};
}
// #endregion

// #region stream
using Stream = std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)>;

Stream make_stream() {
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) {
        s = nullptr;
    }
    return Stream{s, &cudaStreamDestroy};
}
// #endregion

// #region kernel
constexpr int kIters = 300;

__host__ __device__ float crunch(float v) {
    for (int k = 0; k < kIters; ++k) {
        v = sinf(v) + 1.0f;
    }
    return v;
}

__global__ void heavy(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = crunch(x[i]);
    }
}
// #endregion

// #region time_ms
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
// #endregion

// #region pipeline
// Split the work into `chunks` pieces. Chunk c goes to stream c % streams.
void run_pipeline(const float* in, float* out, float* dev, int n, int chunks,
                  const std::vector<Stream>& streams) {
    int chunk = n / chunks;  // n is a multiple of chunks
    std::size_t bytes = chunk * sizeof(float);
    for (int c = 0; c < chunks; ++c) {
        cudaStream_t s = streams[c % streams.size()].get();
        int offset = c * chunk;
        cudaMemcpyAsync(dev + offset, in + offset, bytes,
                        cudaMemcpyHostToDevice, s);
        heavy<<<(chunk + 255) / 256, 256, 0, s>>>(dev + offset, chunk);
        cudaMemcpyAsync(out + offset, dev + offset, bytes,
                        cudaMemcpyDeviceToHost, s);
    }
}
// #endregion

int main() {
    // #region alloc
    constexpr int n = 64 * 1024 * 1024;  // 64M floats = 256 MiB
    constexpr std::size_t bytes = n * sizeof(float);

    std::vector<float> pageable(n);
    auto in = make_pinned_buffer(n);
    auto out = make_pinned_buffer(n);
    auto dev = make_device_buffer(n);
    if (!in || !out || !dev) {
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        in.get()[i] = (i % 1000) * 0.001f;
    }
    // #endregion

    // #region bandwidth
    float pageable_ms = time_ms([&] {
        cudaMemcpy(dev.get(), pageable.data(), bytes, cudaMemcpyHostToDevice);
    }, 1, 5);
    float pinned_ms = time_ms([&] {
        cudaMemcpy(dev.get(), in.get(), bytes, cudaMemcpyHostToDevice);
    }, 1, 5);
    CUDA_CHECK(cudaGetLastError());
    std::printf("H2D %zu MiB, pageable: %6.2f ms, %5.1f GB/s\n",
                bytes >> 20, pageable_ms, bytes / pageable_ms / 1e6);
    std::printf("H2D %zu MiB, pinned:   %6.2f ms, %5.1f GB/s\n",
                bytes >> 20, pinned_ms, bytes / pinned_ms / 1e6);
    // #endregion

    // #region streams
    constexpr int chunks = 16;
    for (int num_streams : {1, 4}) {
        std::vector<Stream> streams;
        for (int s = 0; s < num_streams; ++s) {
            streams.push_back(make_stream());
            if (!streams.back()) {
                return 1;
            }
        }
        float ms = time_ms([&] {
            run_pipeline(in.get(), out.get(), dev.get(), n, chunks, streams);
        }, 1, 5);
        CUDA_CHECK(cudaGetLastError());
        std::printf("%d stream(s), %d chunks: %6.2f ms\n",
                    num_streams, chunks, ms);
    }
    // #endregion

    // #region verify
    int errors = 0;
    for (int i = 0; i < n; i += 997) {
        if (std::fabs(out.get()[i] - crunch(in.get()[i])) > 1e-4f) {
            ++errors;
        }
    }
    std::printf("%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    // #endregion
    return errors == 0 ? 0 : 1;
}
