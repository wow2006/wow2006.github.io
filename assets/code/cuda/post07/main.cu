#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>
#include <cublas_v2.h>
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

// #region naive
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
// #endregion

// #region tiled
constexpr int TILE = 32;

// Assumes n is a multiple of TILE.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < n; t += TILE) {
        // Each thread loads one element of each tile.
        As[ty][tx] = A[row * n + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * n + col];
        __syncthreads();  // tiles are complete

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // everyone is done before we overwrite the tiles
    }
    C[row * n + col] = sum;
}
// #endregion

// #region timer
// Average time of one launch in ms, after one warm-up launch.
template <typename Launch>
float time_ms(Launch launch, int reps) {
    launch();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) {
        launch();
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

// #region verify
// Largest |x - ref| over all elements.
float max_abs_error(const std::vector<float>& x, const std::vector<float>& ref) {
    float err = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        err = std::fmax(err, std::fabs(x[i] - ref[i]));
    }
    return err;
}
// #endregion

int main() {
    // #region host
    constexpr int n = 4096;
    constexpr std::size_t count = std::size_t{n} * n;
    constexpr std::size_t bytes = count * sizeof(float);
    constexpr double flops = 2.0 * n * n * n;  // n*n outputs, n multiply-adds each
    constexpr double peak_gflops = 82'600.0;   // RTX 4090 FP32 peak
    constexpr int reps = 10;
    static_assert(n % TILE == 0, "matmul_tiled needs n to be a multiple of TILE");

    std::vector<float> a(count), b(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    // #endregion

    auto d_a = make_device_buffer(count);
    auto d_b = make_device_buffer(count);
    auto d_c = make_device_buffer(count);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice));

    // #region cublas
    cublasHandle_t raw_handle = nullptr;
    if (cublasCreate(&raw_handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    std::unique_ptr<cublasContext, decltype(&cublasDestroy)> handle{raw_handle, &cublasDestroy};

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS is column-major. Our row-major A, B, C look like A^T, B^T, C^T to it.
    // C = A * B  <=>  C^T = B^T * A^T, so we pass B first, then A.
    auto run_cublas = [&] {
        return cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha, d_b.get(), n, d_a.get(), n,
                           &beta, d_c.get(), n);
    };
    // #endregion

    std::vector<float> ref(count), c(count);
    if (run_cublas() != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm failed\n");
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(ref.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
    float cublas_ms = time_ms(run_cublas, reps);
    CUDA_CHECK(cudaGetLastError());

    // #region launch
    dim3 threads(TILE, TILE);  // 32 x 32 = 1024 threads per block
    dim3 blocks(n / TILE, n / TILE);

    float naive_ms = time_ms([&] {
        matmul_naive<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
    }, reps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
    float naive_err = max_abs_error(c, ref);

    float tiled_ms = time_ms([&] {
        matmul_tiled<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
    }, reps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));
    float tiled_err = max_abs_error(c, ref);
    // #endregion

    // #region report
    std::printf("SGEMM %d x %d x %d, %d runs each\n\n", n, n, n, reps);
    std::printf("%-8s %9s %10s %8s %10s %10s\n",
                "kernel", "time ms", "GFLOP/s", "% peak", "% cuBLAS", "max err");
    auto row = [&](const char* name, float ms, float err) {
        double gflops = flops / (ms * 1e6);
        double cublas_gflops = flops / (cublas_ms * 1e6);
        std::printf("%-8s %9.2f %10.0f %7.1f%% %9.1f%% %10.2e\n", name, ms, gflops,
                    100.0 * gflops / peak_gflops, 100.0 * gflops / cublas_gflops, err);
    };
    row("naive", naive_ms, naive_err);
    row("tiled", tiled_ms, tiled_err);
    row("cuBLAS", cublas_ms, 0.0f);

    constexpr float tolerance = 1e-3f;
    bool ok = naive_err < tolerance && tiled_err < tolerance;
    std::printf("\n%s (tolerance %.0e)\n", ok ? "PASSED" : "FAILED", tolerance);
    // #endregion
    return ok ? 0 : 1;
}
