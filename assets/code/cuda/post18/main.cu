#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_pipeline.h>
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

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "%s:%d: cuBLAS error %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(st));                  \
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

// All kernels compute C = A * B for square, row-major n x n matrices.
// To keep them short, they assume n is a multiple of 128.

// #region naive
// Step 1: one thread per element of C. threadIdx.x walks down a column.
__global__ void sgemm_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}
// #endregion

// #region coalesced
// Step 2: same code, but threadIdx.x now walks along a row.
__global__ void sgemm_coalesced(const float* A, const float* B, float* C,
                                int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}
// #endregion

// #region smem
// Step 3: the shared-memory tiled kernel from post 07.
constexpr int TILE = 32;

__global__ void sgemm_smem(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < n; t += TILE) {
        As[ty][tx] = A[row * n + t + tx];
        Bs[ty][tx] = B[(t + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}
// #endregion

// #region tile1d
// Step 4: each thread computes TM results in one column of a BM x BN tile.
namespace t1d {
constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
constexpr int THREADS = BM * BN / TM;  // 512
}

__global__ void __launch_bounds__(t1d::THREADS)
sgemm_1d(const float* A, const float* B, float* C, int n) {
    using namespace t1d;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Move the pointers to this block's tile.
    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    int tid = threadIdx.x;
    int tcol = tid % BN;       // which column of the tile
    int trow = tid / BN * TM;  // first of our TM rows
    // 512 threads load a 64x8 A tile and an 8x64 B tile, one float each.
    int aRow = tid / BK, aCol = tid % BK;
    int bRow = tid / BN, bCol = tid % BN;

    float acc[TM] = {};
    for (int t = 0; t < n; t += BK) {
        As[aRow][aCol] = A[aRow * n + aCol];
        Bs[bRow][bCol] = B[bRow * n + bCol];
        __syncthreads();
        A += BK;
        B += BK * n;

        for (int k = 0; k < BK; ++k) {
            float b = Bs[k][tcol];  // one shared load ...
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[trow + i][k] * b;  // ... reused TM times
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        C[(trow + i) * n + tcol] = acc[i];
    }
}
// #endregion

// #region tile2d
// Step 5: each thread computes a TM x TN block of results.
namespace t2d {
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
}

__global__ void __launch_bounds__(t2d::THREADS)
sgemm_2d(const float* A, const float* B, float* C, int n) {
    using namespace t2d;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    int tid = threadIdx.x;
    int tcol = tid % (BN / TN) * TN;
    int trow = tid / (BN / TN) * TM;

    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int t = 0; t < n; t += BK) {
        // 256 threads load 1024 floats of each tile: 4 each.
        for (int i = tid; i < BM * BK; i += THREADS) {
            As[i / BK][i % BK] = A[i / BK * n + i % BK];
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            Bs[i / BN][i % BN] = B[i / BN * n + i % BN];
        }
        __syncthreads();
        A += BK;
        B += BK * n;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) a[i] = As[trow + i][k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tcol + j];
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];  // outer product
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(trow + i) * n + tcol + j] = acc[i][j];
        }
    }
}
// #endregion

// #region vec
// Step 6: 128-bit loads and stores; A is stored transposed in shared memory.
__global__ void __launch_bounds__(t2d::THREADS)
sgemm_vec(const float* A, const float* B, float* C, int n) {
    using namespace t2d;
    __shared__ float As[BK][BM];  // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    int tid = threadIdx.x;
    int tcol = tid % (BN / TN) * TN;
    int trow = tid / (BN / TN) * TM;
    // One float4 of A (128x8) and one of B (8x128) per thread.
    int aRow = tid / (BK / 4), aCol = tid % (BK / 4) * 4;
    int bRow = tid / (BN / 4), bCol = tid % (BN / 4) * 4;

    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int t = 0; t < n; t += BK) {
        float4 va = *reinterpret_cast<const float4*>(&A[aRow * n + aCol]);
        As[aCol + 0][aRow] = va.x;
        As[aCol + 1][aRow] = va.y;
        As[aCol + 2][aRow] = va.z;
        As[aCol + 3][aRow] = va.w;
        *reinterpret_cast<float4*>(&Bs[bRow][bCol]) =
            *reinterpret_cast<const float4*>(&B[bRow * n + bCol]);
        __syncthreads();
        A += BK;
        B += BK * n;

        for (int k = 0; k < BK; ++k) {
            // Two 128-bit shared loads each for a[] and b[].
            *reinterpret_cast<float4*>(&a[0]) = *reinterpret_cast<const float4*>(&As[k][trow]);
            *reinterpret_cast<float4*>(&a[4]) = *reinterpret_cast<const float4*>(&As[k][trow + 4]);
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&Bs[k][tcol]);
            *reinterpret_cast<float4*>(&b[4]) = *reinterpret_cast<const float4*>(&Bs[k][tcol + 4]);
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            *reinterpret_cast<float4*>(&C[(trow + i) * n + tcol + j]) =
                *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}
// #endregion

// Step 7: double buffering. While we compute on one pair of tiles, cp.async
// copies the next pair straight from global to shared memory.
namespace tas {
constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
}

__global__ void __launch_bounds__(tas::THREADS)
sgemm_async(const float* A, const float* B, float* C, int n) {
    using namespace tas;
    // cp.async copies 16 bytes as they are, so A is no longer transposed.
    __shared__ alignas(16) float As[2][BM][BK];
    __shared__ alignas(16) float Bs[2][BK][BN];

    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    int tid = threadIdx.x;
    int tcol = tid % (BN / TN) * TN;
    int trow = tid / (BN / TN) * TM;

    // #region load-tile
    // Start copying the tiles that begin at column t of A and row t of B.
    auto load_tile = [&](int buf, int t) {
        for (int i = tid; i < BM * BK / 4; i += THREADS) {
            int r = i / (BK / 4), c = i % (BK / 4) * 4;
            __pipeline_memcpy_async(&As[buf][r][c], &A[r * n + t + c], 16);
        }
        for (int i = tid; i < BK * BN / 4; i += THREADS) {
            int r = i / (BN / 4), c = i % (BN / 4) * 4;
            __pipeline_memcpy_async(&Bs[buf][r][c], &B[(t + r) * n + c], 16);
        }
        __pipeline_commit();
    };
    // #endregion

    float acc[TM][TN] = {};
    float a[TM], b[TN];
    // #region pipeline
    load_tile(0, 0);
    for (int t = 0; t < n; t += BK) {
        int buf = (t / BK) % 2;
        if (t + BK < n) {
            load_tile(buf ^ 1, t + BK);  // start the next copy ...
            __pipeline_wait_prior(1);    // ... then wait only for this one
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) a[i] = As[buf][trow + i][k];
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&Bs[buf][k][tcol]);
            *reinterpret_cast<float4*>(&b[4]) = *reinterpret_cast<const float4*>(&Bs[buf][k][tcol + 4]);
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();  // everyone is done with buf before it is refilled
    }
    // #endregion
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            *reinterpret_cast<float4*>(&C[(trow + i) * n + tcol + j]) =
                *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}

// Step 8: same as step 7, but threads are mapped to outputs warp by warp.
// Each warp owns a 32x64 piece of the tile. Inside it, a thread owns
// 8 rows (row0 + 4*i) and 8 columns (col0..col0+3 and col0+32..col0+35).
namespace twp {
constexpr int BM = 128, BN = 128, BK = 16, PAD = 4;
constexpr int THREADS = 256;
}

__global__ void __launch_bounds__(twp::THREADS)
sgemm_warp(const float* A, const float* B, float* C, int n) {
    using namespace twp;
    __shared__ alignas(16) float As[2][BM][BK + PAD];  // padded rows
    __shared__ alignas(16) float Bs[2][BK][BN];

    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    int tid = threadIdx.x;
    // #region warp-map
    int warp = tid / 32, lane = tid % 32;
    int row0 = (warp / 2) * 32 + lane / 8;       // 4 warps down
    int col0 = (warp % 2) * 64 + lane % 8 * 4;   // 2 warps across
    // #endregion

    auto load_tile = [&](int buf, int t) {
        for (int i = tid; i < BM * BK / 4; i += THREADS) {
            int r = i / (BK / 4), c = i % (BK / 4) * 4;
            __pipeline_memcpy_async(&As[buf][r][c], &A[r * n + t + c], 16);
        }
        for (int i = tid; i < BK * BN / 4; i += THREADS) {
            int r = i / (BN / 4), c = i % (BN / 4) * 4;
            __pipeline_memcpy_async(&Bs[buf][r][c], &B[(t + r) * n + c], 16);
        }
        __pipeline_commit();
    };

    float acc[8][8] = {};
    float a[8], b[8];
    load_tile(0, 0);
    for (int t = 0; t < n; t += BK) {
        int buf = (t / BK) % 2;
        if (t + BK < n) {
            load_tile(buf ^ 1, t + BK);
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        // #region warp-inner
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < 8; ++i) a[i] = As[buf][row0 + 4 * i][k];
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&Bs[buf][k][col0]);
            *reinterpret_cast<float4*>(&b[4]) = *reinterpret_cast<const float4*>(&Bs[buf][k][col0 + 32]);
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        // #endregion
        __syncthreads();
    }
    for (int i = 0; i < 8; ++i) {
        float* c = &C[(row0 + 4 * i) * n + col0];
        *reinterpret_cast<float4*>(c) = *reinterpret_cast<float4*>(&acc[i][0]);
        *reinterpret_cast<float4*>(c + 32) = *reinterpret_cast<float4*>(&acc[i][4]);
    }
}

// #region launch
// Launch kernel `id` (1..8) with the grid and block shape it expects.
void launch(int id, const float* A, const float* B, float* C, int n) {
    switch (id) {
    case 1: sgemm_naive<<<dim3(n / 32, n / 32), dim3(32, 32)>>>(A, B, C, n); break;
    case 2: sgemm_coalesced<<<dim3(n / 32, n / 32), dim3(32, 32)>>>(A, B, C, n); break;
    case 3: sgemm_smem<<<dim3(n / TILE, n / TILE), dim3(TILE, TILE)>>>(A, B, C, n); break;
    case 4: sgemm_1d<<<dim3(n / t1d::BN, n / t1d::BM), t1d::THREADS>>>(A, B, C, n); break;
    case 5: sgemm_2d<<<dim3(n / t2d::BN, n / t2d::BM), t2d::THREADS>>>(A, B, C, n); break;
    case 6: sgemm_vec<<<dim3(n / t2d::BN, n / t2d::BM), t2d::THREADS>>>(A, B, C, n); break;
    case 7: sgemm_async<<<dim3(n / tas::BN, n / tas::BM), tas::THREADS>>>(A, B, C, n); break;
    case 8: sgemm_warp<<<dim3(n / twp::BN, n / twp::BM), twp::THREADS>>>(A, B, C, n); break;
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

constexpr const char* NAMES[] = {"cuBLAS",     "1 naive",   "2 coalesced",
                                 "3 smem tile", "4 1D tile", "5 2D tile",
                                 "6 float4",   "7 cp.async", "8 warp tile"};
constexpr int NUM_KERNELS = 8;

int main(int argc, char** argv) {
    // Optional argument: run only one kernel (handy for ncu).
    int only = argc > 1 ? std::atoi(argv[1]) : -1;
    constexpr int n = 4096;
    constexpr std::size_t count = std::size_t{n} * n;
    constexpr double flops = 2.0 * n * n * n;

    std::vector<float> a(count), b(count), c(count), ref(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(gen);
    for (auto& x : b) x = dist(gen);

    auto d_a = make_device_buffer(count);
    auto d_b = make_device_buffer(count);
    auto d_c = make_device_buffer(count);
    if (!d_a || !d_b || !d_c) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    // #region cublas
    cublasHandle_t raw = nullptr;
    CUBLAS_CHECK(cublasCreate(&raw));
    std::unique_ptr<cublasContext, decltype(&cublasDestroy)> handle{raw, &cublasDestroy};
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. A row-major matrix is its transpose in
    // column-major, so we ask for C^T = B^T * A^T by swapping A and B.
    auto run_cublas = [&] {
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                    d_b.get(), n, d_a.get(), n, &beta, d_c.get(), n);
    };
    // #endregion

    run_cublas();
    CUDA_CHECK(cudaMemcpy(ref.data(), d_c.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
    float cublas_ms = time_ms(run_cublas);
    CUDA_CHECK(cudaGetLastError());

    std::printf("SGEMM %d x %d x %d (FP32)\n", n, n, n);
    std::printf("%-12s %9s %9s %9s %10s\n", "kernel", "ms", "GFLOP/s", "cuBLAS", "max error");
    std::printf("%-12s %9.3f %9.0f %8.1f%% %10s\n", NAMES[0], cublas_ms,
                flops / cublas_ms * 1e-6, 100.0, "-");

    // #region verify
    for (int id = 1; id <= NUM_KERNELS; ++id) {
        if (only > 0 && id != only) continue;
        CUDA_CHECK(cudaMemset(d_c.get(), 0, count * sizeof(float)));
        float ms = time_ms([&] { launch(id, d_a.get(), d_b.get(), d_c.get(), n); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(c.data(), d_c.get(), count * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            max_err = std::fmax(max_err, std::fabs(c[i] - ref[i]));
        }
        bool ok = max_err < 1e-3f;
        std::printf("%-12s %9.3f %9.0f %8.1f%% %10.2e%s\n", NAMES[id], ms,
                    flops / ms * 1e-6, 100.0 * cublas_ms / ms,
                    max_err, ok ? "" : "  FAILED");
        if (!ok) return 1;
    }
    // #endregion
    return 0;
}
