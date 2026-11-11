#include <cstdio>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err));                         \
            return 1;                                                      \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cublasGetStatusString(st));                       \
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

// #region add
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
// #endregion

// #region wmma
// Same WMMA kernel as post 17, but B is now row-major (K x N) like in the
// Python versions. Each warp computes a 32x32 tile of C; a block of 2x2 warps
// covers 64x64. M and N must be multiples of 64, K a multiple of 16.
using namespace nvcuda;
constexpr int WARP_TILE = 32;
constexpr int BLOCK_TILE = 64;

__global__ void gemm_wmma(const half* A, const half* B, float* C,
                          int M, int N, int K) {
    int warp_row = blockIdx.y * BLOCK_TILE + threadIdx.y * WARP_TILE;
    int warp_col = blockIdx.x * BLOCK_TILE + (threadIdx.x / 32) * WARP_TILE;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a[i], A + (warp_row + i * 16) * K + k, K);
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b[j], B + k * N + warp_col + j * 16, N);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(C + (warp_row + i * 16) * N + warp_col + j * 16,
                                    acc[i][j], N, wmma::mem_row_major);
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
    bool ok = true;

    // #region vector-add
    constexpr int n = 1 << 26;  // 64M floats, 256 MB per vector
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i % 1000;
        b[i] = 2 * (i % 1000);
    }
    auto d_a = make_device_buffer<float>(n);
    auto d_b = make_device_buffer<float>(n);
    auto d_c = make_device_buffer<float>(n);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int threads = 1024;
    constexpr int blocks = (n + threads - 1) / threads;
    float add_ms = time_ms([&] {
        add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), n);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    int add_errors = 0;
    for (int i = 0; i < n; ++i) {
        add_errors += c[i] != a[i] + b[i];
    }
    double add_gbs = 3.0 * n * sizeof(float) / (add_ms * 1e-3) / 1e9;
    std::printf("vector add   n = %d  %7.3f ms  %6.1f GB/s  errors %d\n",
                n, add_ms, add_gbs, add_errors);
    ok = ok && add_errors == 0;
    // #endregion

    // #region matmul
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr double flops = 2.0 * M * N * K;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> ha(M * K), hb(K * N);  // both row-major
    for (auto& x : ha) x = __float2half(dist(rng));
    for (auto& x : hb) x = __float2half(dist(rng));

    auto d_ha = make_device_buffer<half>(ha.size());
    auto d_hb = make_device_buffer<half>(hb.size());
    auto d_hc = make_device_buffer<float>(M * N);
    if (!d_ha || !d_hb || !d_hc) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_ha.get(), ha.data(), ha.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hb.get(), hb.data(), hb.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A * B is column-major C^T = B^T * A^T: swap A and B.
    auto run_cublas = [&] {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     d_hb.get(), CUDA_R_16F, N, d_ha.get(), CUDA_R_16F, K, &beta,
                     d_hc.get(), CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    auto run_wmma = [&] {
        gemm_wmma<<<dim3(N / BLOCK_TILE, M / BLOCK_TILE), dim3(64, 2)>>>(
            d_ha.get(), d_hb.get(), d_hc.get(), M, N, K);
    };

    std::vector<float> ref(M * N), out(M * N);
    run_cublas();
    CUDA_CHECK(cudaMemcpy(ref.data(), d_hc.get(), ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double cpu_err = 0.0;  // cuBLAS vs CPU on a few elements
    for (int s = 0; s < 256; ++s) {
        int m = rng() % M, j = rng() % N;
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += double(__half2float(ha[m * K + k])) * __half2float(hb[k * N + j]);
        cpu_err = std::fmax(cpu_err, std::fabs(sum - ref[m * N + j]));
    }
    CUDA_CHECK(cudaMemset(d_hc.get(), 0, M * N * sizeof(float)));
    run_wmma();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_hc.get(), out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float wmma_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        wmma_err = std::fmax(wmma_err, std::fabs(out[i] - ref[i]));

    float cublas_ms = time_ms(run_cublas);
    float wmma_ms = time_ms(run_wmma);
    CUDA_CHECK(cudaGetLastError());
    std::printf("matmul %dx%dx%d, FP16 in, FP32 out\n", M, N, K);
    std::printf("  cuBLAS     %7.3f ms  %6.1f TFLOP/s  max abs err vs CPU    %.2e\n",
                cublas_ms, flops / (cublas_ms * 1e-3) / 1e12, cpu_err);
    std::printf("  WMMA       %7.3f ms  %6.1f TFLOP/s  max abs diff vs cuBLAS %.2e\n",
                wmma_ms, flops / (wmma_ms * 1e-3) / 1e12, wmma_err);
    ok = ok && cpu_err < 1e-2 && wmma_err < 1e-2;
    cublasDestroy(handle);
    // #endregion

    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
