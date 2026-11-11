#include <cstdint>
#include <cstdio>
#include <cmath>
#include <functional>
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

// Each warp computes a 32x32 tile of C; a block of 2x2 warps covers 64x64.
// M and N must be multiples of 64, K a multiple of 16.
constexpr int WARP_TILE = 32;
constexpr int BLOCK_TILE = 64;

// #region wmma
using namespace nvcuda;

// C (MxN, row-major, float) = A (MxK, row-major, half) * B (KxN, column-major, half)
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a[i], A + (warp_row + i * 16) * K + k, K);
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b[j], B + (warp_col + j * 16) * K + k, K);
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

// #region mma-ptx
// d += a * b for one 16x8x16 tile. a: 4 regs (8 halves), b: 2 regs (4 halves),
// d: 4 floats. Each of the 32 lanes passes its own share.
__device__ void mma_16x8x16(float (&d)[4], const uint32_t (&a)[4],
                            const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two consecutive halves as one 32-bit register.
__device__ uint32_t load_pair(const half* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
// #endregion

// #region mma-kernel
__global__ void gemm_mma(const half* A, const half* B, float* C,
                         int M, int N, int K) {
    int warp_row = blockIdx.y * BLOCK_TILE + threadIdx.y * WARP_TILE;
    int warp_col = blockIdx.x * BLOCK_TILE + (threadIdx.x / 32) * WARP_TILE;
    int lane = threadIdx.x % 32;
    int g = lane / 4;  // "group": row of A, column of B and C
    int t = lane % 4;  // thread in group: picks a pair of k (or n)

    float acc[2][4][4] = {};  // 2 tiles of 16 rows x 4 tiles of 8 columns
    for (int k = 0; k < K; k += 16) {
        uint32_t a[2][4], b[4][2];
        for (int i = 0; i < 2; ++i) {
            const half* p = A + (warp_row + i * 16 + g) * K + k + 2 * t;
            a[i][0] = load_pair(p);              // row g,   k 2t..2t+1
            a[i][1] = load_pair(p + 8 * K);      // row g+8, k 2t..2t+1
            a[i][2] = load_pair(p + 8);          // row g,   k 2t+8..2t+9
            a[i][3] = load_pair(p + 8 * K + 8);  // row g+8, k 2t+8..2t+9
        }
        for (int j = 0; j < 4; ++j) {
            const half* p = B + (warp_col + j * 8 + g) * K + k + 2 * t;
            b[j][0] = load_pair(p);      // column g, k 2t..2t+1
            b[j][1] = load_pair(p + 8);  // column g, k 2t+8..2t+9
        }
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                mma_16x8x16(acc[i][j], a[i], b[j]);
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = warp_row + i * 16 + g;
            int col = warp_col + j * 8 + 2 * t;
            C[row * N + col] = acc[i][j][0];
            C[row * N + col + 1] = acc[i][j][1];
            C[(row + 8) * N + col] = acc[i][j][2];
            C[(row + 8) * N + col + 1] = acc[i][j][3];
        }
    }
}
// #endregion

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

float max_abs_diff(const std::vector<float>& x, const std::vector<float>& y) {
    float m = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::fmax(m, std::fabs(x[i] - y[i]));
    return m;
}

int main() {
    // #region host
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr double flops = 2.0 * M * N * K;
    constexpr double peak_tflops = 165.2;  // RTX 4090, FP16 in, FP32 accumulate, dense

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> a(M * K), b(K * N);  // b is column-major: b[n * K + k]
    for (auto& x : a) x = __float2half(dist(rng));
    for (auto& x : b) x = __float2half(dist(rng));
    // #endregion

    auto d_a = make_device_buffer<half>(a.size());
    auto d_b = make_device_buffer<half>(b.size());
    auto d_c = make_device_buffer<float>(M * N);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), a.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), b.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    // #region cublas
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. Our row-major C (MxN) is a column-major C^T (NxM),
    // so we ask for C^T = B^T * A^T, which needs no copies at all.
    auto run_cublas = [&] {
        return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                     d_b.get(), CUDA_R_16F, K,
                     d_a.get(), CUDA_R_16F, K, &beta,
                     d_c.get(), CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    // #endregion

    // #region launch
    dim3 threads(64, 2);  // 2x2 warps
    dim3 blocks(N / BLOCK_TILE, M / BLOCK_TILE);
    auto run_wmma = [&] {
        gemm_wmma<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), M, N, K);
    };
    auto run_mma = [&] {
        gemm_mma<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), M, N, K);
    };
    // #endregion

    // #region verify
    // cuBLAS is the reference. First check it against the CPU on a few elements.
    std::vector<float> ref(M * N), c(M * N);
    CUBLAS_CHECK(run_cublas());
    CUDA_CHECK(cudaMemcpy(ref.data(), d_c.get(), ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double cpu_err = 0.0;
    for (int s = 0; s < 256; ++s) {
        int m = rng() % M, n = rng() % N;
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += double(__half2float(a[m * K + k])) * __half2float(b[n * K + k]);
        cpu_err = std::fmax(cpu_err, std::fabs(sum - ref[m * N + n]));
    }
    std::printf("cuBLAS vs CPU (256 samples): max abs error %.2e\n", cpu_err);
    // #endregion

    // #region measure
    std::printf("M = N = K = %d, FP16 inputs, FP32 accumulate\n", M);
    bool ok = cpu_err < 1e-2;
    struct Run { const char* name; std::function<void()> f; };
    for (const auto& [name, f] : {Run{"cuBLAS", run_cublas}, Run{"WMMA", run_wmma},
                                  Run{"mma.sync", run_mma}}) {
        CUDA_CHECK(cudaMemset(d_c.get(), 0, M * N * sizeof(float)));
        f();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), c.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err = max_abs_diff(c, ref);
        float ms = time_ms(f);
        double tflops = flops / (ms * 1e-3) / 1e12;
        std::printf("%-9s %7.3f ms  %6.1f TFLOP/s  %5.1f%% of peak  max abs diff %.2e\n",
                    name, ms, tflops, 100.0 * tflops / peak_tflops, err);
        ok = ok && err < 1e-2;
    }
    CUDA_CHECK(cudaGetLastError());
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    // #endregion
    cublasDestroy(handle);
    return ok ? 0 : 1;
}
