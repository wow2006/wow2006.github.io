#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
// #region shim
// Workaround for -U_GNU_SOURCE (see post 00): CUTLASS pulls in <mutex>, which
// calls two glibc functions that are only declared when _GNU_SOURCE is set.
#include <pthread.h>
extern "C" {
int pthread_cond_clockwait(pthread_cond_t*, pthread_mutex_t*, clockid_t,
                           const timespec*);
int pthread_mutex_clocklock(pthread_mutex_t*, clockid_t, const timespec*);
}
// #endregion
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err));                         \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// #region gemm
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,     // A: M x K
    cutlass::half_t, cutlass::layout::ColumnMajor,  // B: K x N
    float, cutlass::layout::RowMajor,               // C: M x N
    float,                                          // accumulator
    cutlass::arch::OpClassTensorOp,                 // use tensor cores
    cutlass::arch::Sm80,                            // mma.sync m16n8k16 (sm_80+)
    cutlass::gemm::GemmShape<128, 128, 32>,         // block tile
    cutlass::gemm::GemmShape<64, 64, 32>,           // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>>;           // one mma.sync
// #endregion

int main() {
    constexpr int M = 4096, N = 4096, K = 4096;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<cutlass::half_t> a(M * K), b(K * N);  // b[n * K + k]
    for (auto& x : a) x = cutlass::half_t(dist(rng));
    for (auto& x : b) x = cutlass::half_t(dist(rng));
    std::vector<float> c(M * N);

    // Raw cudaMalloc/cudaFree keep this side example short (see post 01 for unique_ptr).
    cutlass::half_t *d_a, *d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(a[0])));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(b[0])));
    CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(a[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(b[0]), cudaMemcpyHostToDevice));

    // #region run
    Gemm gemm;
    Gemm::Arguments args({M, N, K},     // problem size
                         {d_a, K},      // A and its leading dimension
                         {d_b, K},      // B
                         {d_c, N},      // C (input, scaled by beta)
                         {d_c, N},      // D (output)
                         {1.0f, 0.0f}); // alpha, beta
    cutlass::Status status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS: %s\n", cutlassGetStatusString(status));
        return 1;
    }
    // #endregion
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double err = 0.0;
    for (int s = 0; s < 256; ++s) {
        int m = rng() % M, n = rng() % N;
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += double(float(a[m * K + k])) * float(b[n * K + k]);
        err = std::fmax(err, std::fabs(sum - c[m * N + n]));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int reps = 20;
    gemm(args);  // warm-up
    cudaEventRecord(start);
    for (int r = 0; r < reps; ++r) gemm(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= reps;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;

    std::printf("CUTLASS   %7.3f ms  %6.1f TFLOP/s  %5.1f%% of peak  max abs error vs CPU %.2e\n",
                ms, tflops, 100.0 * tflops / 165.2, err);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return err < 1e-2 ? 0 : 1;
}
