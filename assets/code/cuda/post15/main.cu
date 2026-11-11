#include <algorithm>
#include <chrono>
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

// #region buffers
using DeviceBuffer = std::unique_ptr<float, decltype(&cudaFree)>;
using PinnedBuffer = std::unique_ptr<float[], decltype(&cudaFreeHost)>;
using Stream = std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)>;

// Allocates on the *current* device (the one chosen with cudaSetDevice).
DeviceBuffer make_device_buffer(std::size_t count) {
    float* ptr = nullptr;
    if (cudaMalloc(&ptr, count * sizeof(float)) != cudaSuccess) {
        ptr = nullptr;
    }
    return DeviceBuffer{ptr, &cudaFree};
}

PinnedBuffer make_pinned_buffer(std::size_t count) {
    float* ptr = nullptr;
    if (cudaMallocHost(&ptr, count * sizeof(float)) != cudaSuccess) {
        ptr = nullptr;
    }
    return PinnedBuffer{ptr, &cudaFreeHost};
}

// Creates a stream on the *current* device.
Stream make_stream() {
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) {
        s = nullptr;
    }
    return Stream{s, &cudaStreamDestroy};
}
// #endregion

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Adds a + b into c using the first `ngpu` GPUs, each one taking a slice.
// Returns the average time of one full run (copy in, add, copy out) in `ms`.
int split_add(int ngpu, const float* a, const float* b, float* c, int n,
              double& ms) {
    // #region slices
    struct Slice {
        DeviceBuffer a, b, c;
        Stream stream;
        int offset, count;
    };
    std::vector<Slice> slices;

    int per_gpu = (n + ngpu - 1) / ngpu;
    for (int dev = 0; dev < ngpu; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        int offset = dev * per_gpu;
        int count = std::min(per_gpu, n - offset);
        slices.push_back({make_device_buffer(count), make_device_buffer(count),
                          make_device_buffer(count), make_stream(), offset,
                          count});
        const Slice& s = slices.back();
        if (!s.a || !s.b || !s.c || !s.stream) {
            return 1;
        }
    }
    // #endregion

    constexpr int reps = 10;
    Clock::time_point t0;
    for (int rep = 0; rep <= reps; ++rep) {  // rep 0 is the warm-up
        if (rep == 1) {
            t0 = Clock::now();
        }
        // #region launch
        for (int dev = 0; dev < ngpu; ++dev) {
            const Slice& s = slices[dev];
            std::size_t bytes = s.count * sizeof(float);
            CUDA_CHECK(cudaSetDevice(dev));
            CUDA_CHECK(cudaMemcpyAsync(s.a.get(), a + s.offset, bytes,
                                       cudaMemcpyHostToDevice, s.stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(s.b.get(), b + s.offset, bytes,
                                       cudaMemcpyHostToDevice, s.stream.get()));
            int threads = 256;
            int blocks = (s.count + threads - 1) / threads;
            add<<<blocks, threads, 0, s.stream.get()>>>(s.a.get(), s.b.get(),
                                                        s.c.get(), s.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(c + s.offset, s.c.get(), bytes,
                                       cudaMemcpyDeviceToHost, s.stream.get()));
        }
        for (int dev = 0; dev < ngpu; ++dev) {
            CUDA_CHECK(cudaSetDevice(dev));
            CUDA_CHECK(cudaStreamSynchronize(slices[dev].stream.get()));
        }
        // #endregion
    }
    ms = ms_since(t0) / reps;
    return 0;
}

int main() {
    // #region count
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::printf("Found %d GPU(s)\n", count);
    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("  GPU %d: %s, PCI bus %02x\n", dev, prop.name,
                    prop.pciBusID);
    }
    // #endregion

    // #region host
    constexpr int n = 1 << 26;  // 64M floats = 256 MiB per array
    auto a = make_pinned_buffer(n);
    auto b = make_pinned_buffer(n);
    auto c = make_pinned_buffer(n);
    if (!a || !b || !c) {
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        a[i] = i % 1000;
        b[i] = 2 * (i % 1000);
    }
    // #endregion

    // #region compare
    for (int ngpu = 1; ngpu <= count; ++ngpu) {
        double ms = 0;
        std::fill(c.get(), c.get() + n, 0.0f);
        if (split_add(ngpu, a.get(), b.get(), c.get(), n, ms) != 0) {
            return 1;
        }
        int errors = 0;
        for (int i = 0; i < n; ++i) {
            if (c[i] != 3.0f * (i % 1000)) {
                ++errors;
            }
        }
        std::printf("%d GPU(s): %6.2f ms per run  %s (%d errors)\n", ngpu, ms,
                    errors == 0 ? "PASSED" : "FAILED", errors);
    }
    // #endregion

    if (count < 2) {
        std::printf("Only one GPU, skipping the peer-to-peer part\n");
        return 0;
    }

    // #region can-access
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    std::printf("Can GPU 0 access GPU 1 memory? %s\n", can01 ? "yes" : "no");
    std::printf("Can GPU 1 access GPU 0 memory? %s\n", can10 ? "yes" : "no");

    if (can01 && can10) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
        std::printf("Peer access enabled in both directions\n");
    }
    // #endregion

    // #region peer-copy
    CUDA_CHECK(cudaSetDevice(0));
    auto src = make_device_buffer(n);
    CUDA_CHECK(cudaSetDevice(1));
    auto dst = make_device_buffer(n);
    if (!src || !dst) {
        return 1;
    }
    constexpr std::size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(src.get(), a.get(), bytes, cudaMemcpyHostToDevice));

    constexpr int reps = 10;
    Clock::time_point t0;
    for (int rep = 0; rep <= reps; ++rep) {  // rep 0 is the warm-up
        if (rep == 1) {
            t0 = Clock::now();
        }
        CUDA_CHECK(cudaMemcpyPeer(dst.get(), 1, src.get(), 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());  // current device is 0
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaSetDevice(0));
    }
    double peer_ms = ms_since(t0) / reps;
    // #endregion

    // #region staged-copy
    for (int rep = 0; rep <= reps; ++rep) {
        if (rep == 1) {
            t0 = Clock::now();
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(c.get(), src.get(), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(dst.get(), c.get(), bytes, cudaMemcpyHostToDevice));
    }
    double staged_ms = ms_since(t0) / reps;
    // #endregion

    // #region verify-peer
    CUDA_CHECK(cudaMemcpyPeer(dst.get(), 1, src.get(), 0, bytes));
    CUDA_CHECK(cudaMemcpy(c.get(), dst.get(), bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i]) {
            ++errors;
        }
    }
    double gb = bytes / 1e9;
    std::printf("cudaMemcpyPeer GPU 0 -> GPU 1: %6.2f ms  %5.1f GB/s  %s\n",
                peer_ms, gb / (peer_ms / 1e3), errors == 0 ? "PASSED" : "FAILED");
    std::printf("Staged via host (D2H + H2D):   %6.2f ms  %5.1f GB/s\n",
                staged_ms, gb / (staged_ms / 1e3));
    // #endregion
    return errors == 0 ? 0 : 1;
}
