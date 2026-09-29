#include <cstdio>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
// #region include
#include <nvtx3/nvToolsExt.h>
// #endregion

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

int main() {
    constexpr int n = 1'000'000;
    constexpr std::size_t bytes = n * sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    auto d_a = make_device_buffer(n);
    auto d_b = make_device_buffer(n);
    auto d_c = make_device_buffer(n);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }

    // #region copy-in
    nvtxRangePushA("copy-in");
    CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();
    // #endregion

    // #region launch
    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;
    nvtxRangePushA("compute");
    add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
    // #endregion

    // #region copy-out
    nvtxRangePushA("copy-out");
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();
    // #endregion

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f * i) {
            ++errors;
        }
    }
    std::printf("Launched %d blocks x %d threads for %d elements\n",
                blocks, threads, n);
    std::printf("%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    return errors == 0 ? 0 : 1;
}
