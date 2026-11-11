#include <cfloat>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
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

// Average GPU time of one call to `work`, in milliseconds.
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

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;           // threads per block for softmax
constexpr int kWarps = kThreads / 32;

// #region maxsum
// A running softmax state: the largest value seen so far (m) and the sum of
// exp(x - m) over everything seen so far (s).
struct MaxSum {
    float m;
    float s;
};

// Merge two states. Whichever side had the smaller max gets its sum rescaled
// to the new max, so both sums are measured against the same reference.
__device__ MaxSum combine(MaxSum a, MaxSum b) {
    float m = fmaxf(a.m, b.m);
    return {m, a.s * expf(a.m - m) + b.s * expf(b.m - m)};
}
// #endregion

struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Sum {
    __device__ float operator()(float a, float b) const { return a + b; }
};
struct Combine {
    __device__ MaxSum operator()(MaxSum a, MaxSum b) const { return combine(a, b); }
};

// #region reduce
__device__ float shfl_xor(float v, int mask) {
    return __shfl_xor_sync(0xffffffff, v, mask);
}
__device__ MaxSum shfl_xor(MaxSum v, int mask) {
    return {shfl_xor(v.m, mask), shfl_xor(v.s, mask)};
}

// Butterfly reduction: after 5 steps, every lane holds the warp's result.
template <typename T, typename Op>
__device__ T warp_reduce(T v, Op op) {
    for (int mask = 16; mask > 0; mask /= 2) {
        v = op(v, shfl_xor(v, mask));
    }
    return v;
}

// Every thread of the block gets the block's result.
template <typename T, typename Op>
__device__ T block_reduce(T v, Op op) {
    __shared__ T partial[kWarps];
    v = warp_reduce(v, op);
    __syncthreads();                    // an earlier call may still be reading
    if (threadIdx.x % 32 == 0) {
        partial[threadIdx.x / 32] = v;
    }
    __syncthreads();
    v = partial[0];
    for (int w = 1; w < kWarps; ++w) {
        v = op(v, partial[w]);
    }
    return v;
}
// #endregion

// ---------------------------------------------------------------------------
// Part 1: row-wise softmax. One block per row.
// ---------------------------------------------------------------------------

constexpr double kPeakGBs = 1008.0;     // RTX 4090 memory bandwidth

__global__ void copy(const float* x, float* y, std::size_t n) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

// #region naive
__global__ void row_max(const float* x, float* row_m, int cols) {
    const float* row = x + blockIdx.x * std::size_t(cols);
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        m = fmaxf(m, row[j]);
    }
    m = block_reduce(m, Max{});
    if (threadIdx.x == 0) {
        row_m[blockIdx.x] = m;
    }
}

__global__ void row_sum_exp(const float* x, const float* row_m, float* row_s,
                            int cols) {
    const float* row = x + blockIdx.x * std::size_t(cols);
    float m = row_m[blockIdx.x];
    float s = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        s += expf(row[j] - m);
    }
    s = block_reduce(s, Sum{});
    if (threadIdx.x == 0) {
        row_s[blockIdx.x] = s;
    }
}

__global__ void row_normalize(const float* x, const float* row_m,
                              const float* row_s, float* y, int cols) {
    std::size_t offset = blockIdx.x * std::size_t(cols);
    float m = row_m[blockIdx.x];
    float s = row_s[blockIdx.x];
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y[offset + j] = expf(x[offset + j] - m) / s;
    }
}
// #endregion

// #region fused
__global__ void softmax_fused(const float* x, float* y, int cols) {
    std::size_t offset = blockIdx.x * std::size_t(cols);
    const float* row = x + offset;

    float m = -FLT_MAX;                             // pass 1: max
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        m = fmaxf(m, row[j]);
    }
    m = block_reduce(m, Max{});

    float s = 0.0f;                                 // pass 2: sum of exp
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        s += expf(row[j] - m);
    }
    s = block_reduce(s, Sum{});

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {   // pass 3: write
        y[offset + j] = expf(row[j] - m) / s;
    }
}
// #endregion

// #region online
// x and y may be the same buffer (the attention code below uses it in place).
__global__ void softmax_online(const float* x, float* y, int cols) {
    std::size_t offset = blockIdx.x * std::size_t(cols);

    MaxSum state{-FLT_MAX, 0.0f};                   // pass 1: max and sum
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        state = combine(state, {x[offset + j], 1.0f});
    }
    state = block_reduce(state, Combine{});

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {   // pass 2: write
        y[offset + j] = expf(x[offset + j] - state.m) / state.s;
    }
}
// #endregion

int run_softmax() {
    // #region softmax-setup
    constexpr int rows = 8192;
    constexpr int cols = 4096;
    constexpr std::size_t count = std::size_t(rows) * cols;
    constexpr double bytes = 2.0 * count * sizeof(float);   // read x, write y

    std::vector<float> x(count), y(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : x) {
        v = dist(rng);
    }
    // #endregion

    auto d_x = make_device_buffer(count);
    auto d_y = make_device_buffer(count);
    auto d_m = make_device_buffer(rows);
    auto d_s = make_device_buffer(rows);
    if (!d_x || !d_y || !d_m || !d_s) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));

    // CPU reference in double precision.
    std::vector<double> ref(count);
    for (int r = 0; r < rows; ++r) {
        const float* row = &x[std::size_t(r) * cols];
        double m = row[0];
        for (int j = 1; j < cols; ++j) {
            m = std::fmax(m, row[j]);
        }
        double s = 0.0;
        for (int j = 0; j < cols; ++j) {
            s += std::exp(row[j] - m);
        }
        for (int j = 0; j < cols; ++j) {
            ref[std::size_t(r) * cols + j] = std::exp(row[j] - m) / s;
        }
    }

    auto report = [&](const char* name, float ms) {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y.data(), d_y.get(), count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double err = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            err = std::fmax(err, std::fabs(y[i] - ref[i]));
        }
        double gbs = bytes / (ms * 1e6);
        std::printf("%-18s %.3f ms, %4.0f GB/s (%4.1f%% of peak)  max err %.1e\n",
                    name, ms, gbs, 100.0 * gbs / kPeakGBs, err);
        return 0;
    };

    std::printf("Softmax of %d rows x %d cols (%.0f MB in, %.0f MB out)\n",
                rows, cols, count * 4 / 1e6, count * 4 / 1e6);

    // #region softmax-launch
    float copy_ms = time_ms([&] {
        copy<<<(count + kThreads - 1) / kThreads, kThreads>>>(d_x.get(),
                                                              d_y.get(), count);
    });
    CUDA_CHECK(cudaGetLastError());
    double copy_gbs = bytes / (copy_ms * 1e6);
    std::printf("%-18s %.3f ms, %4.0f GB/s (%4.1f%% of peak)\n", "copy (ceiling)",
                copy_ms, copy_gbs, 100.0 * copy_gbs / kPeakGBs);

    float naive_ms = time_ms([&] {
        row_max<<<rows, kThreads>>>(d_x.get(), d_m.get(), cols);
        row_sum_exp<<<rows, kThreads>>>(d_x.get(), d_m.get(), d_s.get(), cols);
        row_normalize<<<rows, kThreads>>>(d_x.get(), d_m.get(), d_s.get(),
                                          d_y.get(), cols);
    });
    if (report("naive (3 kernels)", naive_ms)) return 1;

    float fused_ms = time_ms([&] {
        softmax_fused<<<rows, kThreads>>>(d_x.get(), d_y.get(), cols);
    });
    if (report("fused (3 passes)", fused_ms)) return 1;

    float online_ms = time_ms([&] {
        softmax_online<<<rows, kThreads>>>(d_x.get(), d_y.get(), cols);
    });
    if (report("online (2 passes)", online_ms)) return 1;
    // #endregion
    return 0;
}

// ---------------------------------------------------------------------------
// Part 2: attention O = softmax(Q K^T / sqrt(d)) V, single head, FP32.
// Q, K, V, O are n x d, row-major.
// ---------------------------------------------------------------------------

constexpr int kHeadDim = 64;        // d
constexpr int kTileKeys = 64;       // keys per shared-memory tile
constexpr int kRowsPerBlock = 8;    // query rows per block, one warp each

// #region attn-naive
// S[i][j] = dot(Q[i], K[j]) * scale. One thread per score.
__global__ void attention_scores(const float* q, const float* k, float* s,
                                 int n, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float acc = 0.0f;
        for (int c = 0; c < kHeadDim; ++c) {
            acc += q[i * kHeadDim + c] * k[j * kHeadDim + c];
        }
        s[std::size_t(i) * n + j] = acc * scale;
    }
}

// O[i][c] = sum_j P[i][j] * V[j][c]. One thread per output element.
__global__ void attention_pv(const float* p, const float* v, float* o, int n) {
    int c = threadIdx.x;                            // blockDim.x == kHeadDim
    int i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i < n) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
            acc += p[std::size_t(i) * n + j] * v[j * kHeadDim + c];
        }
        o[i * kHeadDim + c] = acc;
    }
}
// #endregion

// #region flash
// One warp per query row. Each lane owns 2 of the 64 output columns:
// `lane` and `lane + 32`. The N x N score matrix never leaves registers.
// ponytail: assumes n % kTileKeys == 0 and n % kRowsPerBlock == 0.
__global__ void flash_attention(const float* q, const float* k, const float* v,
                                float* o, int n, float scale) {
    __shared__ float k_tile[kTileKeys][kHeadDim];
    __shared__ float v_tile[kTileKeys][kHeadDim];

    int lane = threadIdx.x % 32;
    int row = blockIdx.x * kRowsPerBlock + threadIdx.x / 32;

    float q0 = q[row * kHeadDim + lane] * scale;
    float q1 = q[row * kHeadDim + lane + 32] * scale;
    float o0 = 0.0f;
    float o1 = 0.0f;
    MaxSum state{-FLT_MAX, 0.0f};

    for (int tile = 0; tile < n; tile += kTileKeys) {
        // All warps load the next K and V tiles together.
        __syncthreads();                    // everyone is done with the old tile
        for (int i = threadIdx.x; i < kTileKeys * kHeadDim; i += blockDim.x) {
            k_tile[0][i] = k[tile * kHeadDim + i];
            v_tile[0][i] = v[tile * kHeadDim + i];
        }
        __syncthreads();

        for (int j = 0; j < kTileKeys; ++j) {
            // Score of this query against key j, summed across the warp.
            float s = q0 * k_tile[j][lane] + q1 * k_tile[j][lane + 32];
            s = warp_reduce(s, Sum{});

            // Online softmax: if the max grew, shrink what we have so far.
            float m = fmaxf(state.m, s);
            float correction = expf(state.m - m);
            float p = expf(s - m);
            state = {m, state.s * correction + p};
            o0 = o0 * correction + p * v_tile[j][lane];
            o1 = o1 * correction + p * v_tile[j][lane + 32];
        }
    }
    o[row * kHeadDim + lane] = o0 / state.s;
    o[row * kHeadDim + lane + 32] = o1 / state.s;
}
// #endregion

int run_attention(int n) {
    const std::size_t qkv_count = std::size_t(n) * kHeadDim;
    const std::size_t s_count = std::size_t(n) * n;
    const float scale = 1.0f / std::sqrt(float(kHeadDim));

    std::vector<float> q(qkv_count), k(qkv_count), v(qkv_count);
    std::mt19937 rng(n);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (std::size_t i = 0; i < qkv_count; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
        v[i] = dist(rng);
    }

    auto d_q = make_device_buffer(qkv_count);
    auto d_k = make_device_buffer(qkv_count);
    auto d_v = make_device_buffer(qkv_count);
    auto d_o = make_device_buffer(qkv_count);
    auto d_s = make_device_buffer(s_count);         // only the naive path
    if (!d_q || !d_k || !d_v || !d_o || !d_s) {
        return 1;
    }
    const std::size_t qkv_bytes = qkv_count * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_q.get(), q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k.get(), k.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v.get(), v.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // CPU reference (double) for every 64th query row.
    auto max_error = [&](const std::vector<float>& out) {
        double err = 0.0;
        std::vector<double> p(n);
        for (int i = 0; i < n; i += 64) {
            double m = -1e300;
            for (int j = 0; j < n; ++j) {
                double dot = 0.0;
                for (int c = 0; c < kHeadDim; ++c) {
                    dot += double(q[i * kHeadDim + c]) * k[j * kHeadDim + c];
                }
                p[j] = dot * scale;
                m = std::fmax(m, p[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                p[j] = std::exp(p[j] - m);
                sum += p[j];
            }
            for (int c = 0; c < kHeadDim; ++c) {
                double acc = 0.0;
                for (int j = 0; j < n; ++j) {
                    acc += p[j] * v[j * kHeadDim + c];
                }
                err = std::fmax(err, std::fabs(acc / sum - out[i * kHeadDim + c]));
            }
        }
        return err;
    };
    std::vector<float> out(qkv_count);

    // #region attn-launch
    dim3 score_block(32, 8);
    dim3 score_grid(n / 32, n / 8);
    dim3 pv_block(kHeadDim, 4);
    float scores_ms = time_ms([&] {
        attention_scores<<<score_grid, score_block>>>(d_q.get(), d_k.get(),
                                                      d_s.get(), n, scale);
    });
    float softmax_ms = time_ms([&] {
        softmax_online<<<n, kThreads>>>(d_s.get(), d_s.get(), n);
    });
    float pv_ms = time_ms([&] {
        attention_pv<<<n / 4, pv_block>>>(d_s.get(), d_v.get(), d_o.get(), n);
    });
    CUDA_CHECK(cudaGetLastError());
    // #endregion
    // The softmax ran many times in place above, so run the chain once more
    // from scratch to get a correct result to check.
    attention_scores<<<score_grid, score_block>>>(d_q.get(), d_k.get(),
                                                  d_s.get(), n, scale);
    softmax_online<<<n, kThreads>>>(d_s.get(), d_s.get(), n);
    attention_pv<<<n / 4, pv_block>>>(d_s.get(), d_v.get(), d_o.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_o.get(), qkv_bytes, cudaMemcpyDeviceToHost));
    double naive_err = max_error(out);

    // #region flash-launch
    float flash_ms = time_ms([&] {
        flash_attention<<<n / kRowsPerBlock, kRowsPerBlock * 32>>>(
            d_q.get(), d_k.get(), d_v.get(), d_o.get(), n, scale);
    });
    // #endregion
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_o.get(), qkv_bytes, cudaMemcpyDeviceToHost));
    double flash_err = max_error(out);

    double naive_ms = scores_ms + softmax_ms + pv_ms;
    double flops = 4.0 * n * n * kHeadDim;          // two n x n x d matmuls
    std::printf("n = %d: S matrix %.0f MB, Q/K/V/O %.1f MB\n", n,
                s_count * 4 / 1e6, 4 * qkv_bytes / 1e6);
    std::printf("  naive  %6.3f ms (scores %.3f + softmax %.3f + PV %.3f)"
                "  %5.0f GFLOP/s  max err %.1e\n",
                naive_ms, scores_ms, softmax_ms, pv_ms, flops / (naive_ms * 1e6),
                naive_err);
    std::printf("  flash  %6.3f ms  %5.0f GFLOP/s  max err %.1e\n", flash_ms,
                flops / (flash_ms * 1e6), flash_err);
    return 0;
}

int main() {
    if (run_softmax()) return 1;
    std::printf("\nAttention, single head, d = %d, FP32\n", kHeadDim);
    for (int n : {1024, 2048, 4096}) {
        if (run_attention(n)) return 1;
    }
    return 0;
}
