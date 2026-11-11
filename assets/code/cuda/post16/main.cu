#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>
// #region headers
#include <cuda_fp16.h>   // __half, __half2
#include <cuda_bf16.h>   // __nv_bfloat16
#include <cuda_fp8.h>    // __nv_fp8_e4m3, __nv_fp8_e5m2
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

// ---------------------------------------------------------------------------
// 1. Round trip: float -> smaller format -> float
// ---------------------------------------------------------------------------
// #region round-trip
__global__ void round_trip(const float* in, float* out, int n) {
    int i = threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[4 * i + 0] = __half2float(__float2half(x));
        out[4 * i + 1] = __bfloat162float(__float2bfloat16(x));
        out[4 * i + 2] = float(__nv_fp8_e4m3(x));
        out[4 * i + 3] = float(__nv_fp8_e5m2(x));
    }
}
// #endregion

// ---------------------------------------------------------------------------
// 2. Summing many small numbers: storage type T, accumulator type Acc
// ---------------------------------------------------------------------------
// #region column-sum
template <typename T, typename Acc>
__global__ void column_sum(const T* x, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        Acc acc = Acc(0.0f);
        for (int r = 0; r < rows; ++r) {
            acc += Acc(x[r * cols + col]);
        }
        out[col] = float(acc);
    }
}
// #endregion

// ---------------------------------------------------------------------------
// 3. Bandwidth: out = a * x + y in float and in half2
// ---------------------------------------------------------------------------
// #region axpy
__global__ void axpy_float(int n, float a, const float* x, const float* y,
                           float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

__global__ void axpy_half(int n, __half a, const __half* x, const __half* y,
                          __half* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hfma(a, x[i], y[i]);
    }
}

// n2 = number of __half2 pairs = n / 2
__global__ void axpy_half2(int n2, __half2 a, const __half2* x,
                           const __half2* y, __half2* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        out[i] = __hfma2(a, x[i], y[i]);
    }
}
// #endregion

// ---------------------------------------------------------------------------
// 4. Fast math: the same formula with accurate functions and with intrinsics
// ---------------------------------------------------------------------------
// #region math
constexpr int kTerms = 32;

template <bool Fast>
__global__ void heavy_math(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int k = 1; k <= kTerms; ++k) {
            float t = k * x[i];
            if constexpr (Fast) {
                sum += __fdividef((2.0f + __sinf(t)) * __expf(-0.1f * t),
                                  1.0f + t);
            } else {
                sum += (2.0f + sinf(t)) * expf(-0.1f * t) / (1.0f + t);
            }
        }
        y[i] = sum;
    }
}

double heavy_math_cpu(double x) {
    double sum = 0.0;
    for (int k = 1; k <= kTerms; ++k) {
        double t = k * x;
        sum += (2.0 + std::sin(t)) * std::exp(-0.1 * t) / (1.0 + t);
    }
    return sum;
}
// #endregion

// #region ftz
__global__ void halve(const float* in, float* out) {
    out[0] = in[0] * 0.5f;
}
// #endregion

int main() {
    // ---------------------------------------------------------------------
    // 1. Round trip
    // ---------------------------------------------------------------------
    {
        std::vector<float> values = {0.1f, 1.0f / 3.0f, 3.14159265f, 300.0f,
                                     70000.0f, 1e-5f};
        int n = values.size();
        std::vector<float> back(4 * n);
        auto d_in = make_device_buffer<float>(n);
        auto d_out = make_device_buffer<float>(4 * n);
        if (!d_in || !d_out) {
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_in.get(), values.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));
        round_trip<<<1, n>>>(d_in.get(), d_out.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(back.data(), d_out.get(), 4 * n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        std::printf("== Round trip float -> format -> float ==\n");
        std::printf("%12s %12s %12s %12s %12s\n", "float", "fp16", "bf16",
                    "fp8 e4m3", "fp8 e5m2");
        for (int i = 0; i < n; ++i) {
            std::printf("%12.7g %12.7g %12.7g %12.7g %12.7g\n", values[i],
                        back[4 * i], back[4 * i + 1], back[4 * i + 2],
                        back[4 * i + 3]);
        }
    }

    // ---------------------------------------------------------------------
    // 2. Summing 10,000 copies of 0.001
    // ---------------------------------------------------------------------
    {
        // #region sum-host
        constexpr int rows = 10'000;
        constexpr int cols = 1024;
        std::vector<__half> h(rows * cols, __float2half(0.001f));
        std::vector<__nv_bfloat16> b(rows * cols, __float2bfloat16(0.001f));
        // #endregion
        auto d_h = make_device_buffer<__half>(h.size());
        auto d_b = make_device_buffer<__nv_bfloat16>(b.size());
        auto d_sum = make_device_buffer<float>(cols);
        if (!d_h || !d_b || !d_sum) {
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_h.get(), h.data(), h.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(),
                              b.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));

        std::printf("\n== Sum of %d x 0.001 (exact answer: 10) ==\n", rows);
        float sum = 0.0f;
        // #region sum-launch
        column_sum<__half, __half><<<cols / 256, 256>>>(d_h.get(), d_sum.get(),
                                                        rows, cols);
        CUDA_CHECK(cudaMemcpy(&sum, d_sum.get(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("store fp16, accumulate fp16: %9.4f\n", sum);

        column_sum<__half, float><<<cols / 256, 256>>>(d_h.get(), d_sum.get(),
                                                       rows, cols);
        CUDA_CHECK(cudaMemcpy(&sum, d_sum.get(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("store fp16, accumulate fp32: %9.4f\n", sum);
        // #endregion

        column_sum<__nv_bfloat16, __nv_bfloat16><<<cols / 256, 256>>>(
            d_b.get(), d_sum.get(), rows, cols);
        CUDA_CHECK(cudaMemcpy(&sum, d_sum.get(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("store bf16, accumulate bf16: %9.4f\n", sum);

        column_sum<__nv_bfloat16, float><<<cols / 256, 256>>>(
            d_b.get(), d_sum.get(), rows, cols);
        CUDA_CHECK(cudaMemcpy(&sum, d_sum.get(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("store bf16, accumulate fp32: %9.4f\n", sum);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---------------------------------------------------------------------
    // 3. Bandwidth of float vs half vs half2
    // ---------------------------------------------------------------------
    {
        constexpr int n = 1 << 25;   // 32M elements, must be even for half2
        constexpr int threads = 256;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(n), y(n), out(n);
        std::vector<__half> xh(n), yh(n), outh(n);
        for (int i = 0; i < n; ++i) {
            x[i] = dist(rng);
            y[i] = dist(rng);
            xh[i] = __float2half(x[i]);
            yh[i] = __float2half(y[i]);
        }

        auto d_x = make_device_buffer<float>(n);
        auto d_y = make_device_buffer<float>(n);
        auto d_out = make_device_buffer<float>(n);
        auto d_xh = make_device_buffer<__half>(n);
        auto d_yh = make_device_buffer<__half>(n);
        auto d_outh = make_device_buffer<__half>(n);
        if (!d_x || !d_y || !d_out || !d_xh || !d_yh || !d_outh) {
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y.get(), y.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xh.get(), xh.data(), n * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_yh.get(), yh.data(), n * sizeof(__half),
                              cudaMemcpyHostToDevice));

        // #region axpy-launch
        const float a = 0.5f;
        const __half ah = __float2half(a);
        const __half2 a2 = __half2half2(ah);   // {a, a}
        auto* x2 = reinterpret_cast<const __half2*>(d_xh.get());
        auto* y2 = reinterpret_cast<const __half2*>(d_yh.get());
        auto* out2 = reinterpret_cast<__half2*>(d_outh.get());

        float ms_float = time_ms([&] {
            axpy_float<<<(n + threads - 1) / threads, threads>>>(
                n, a, d_x.get(), d_y.get(), d_out.get());
        });
        float ms_half = time_ms([&] {
            axpy_half<<<(n + threads - 1) / threads, threads>>>(
                n, ah, d_xh.get(), d_yh.get(), d_outh.get());
        });
        float ms_half2 = time_ms([&] {
            axpy_half2<<<(n / 2 + threads - 1) / threads, threads>>>(
                n / 2, a2, x2, y2, out2);
        });
        CUDA_CHECK(cudaGetLastError());
        // #endregion

        CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outh.data(), d_outh.get(), n * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            max_diff = std::fmax(max_diff,
                                 std::fabs(__half2float(outh[i]) - out[i]));
        }

        // #region axpy-score
        constexpr double peak_gbs = 1008.0;   // RTX 4090, from post 03
        auto score = [&](const char* name, float ms, int bytes_per_value) {
            double gbs = 3.0 * n * bytes_per_value / (ms * 1e-3) / 1e9;
            std::printf("%-6s: %.3f ms, %.0f GB/s (%.1f%% of peak)\n", name, ms,
                        gbs, 100.0 * gbs / peak_gbs);
        };
        std::printf("\n== out = a * x + y, %d elements ==\n", n);
        score("float", ms_float, 4);
        score("half", ms_half, 2);
        score("half2", ms_half2, 2);
        // #endregion
        std::printf("max |half2 - float| = %g\n", max_diff);
    }

    // ---------------------------------------------------------------------
    // 4. Fast math: speed and accuracy against a double CPU reference
    // ---------------------------------------------------------------------
    {
        constexpr int n = 1 << 22;
        constexpr int threads = 256;
        constexpr int blocks = (n + threads - 1) / threads;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = 100.0f * i / n;   // x in [0, 100)
        }
        auto d_x = make_device_buffer<float>(n);
        auto d_y = make_device_buffer<float>(n);
        if (!d_x || !d_y) {
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        // #region accuracy
        std::vector<double> ref(n);
        for (int i = 0; i < n; ++i) {
            ref[i] = heavy_math_cpu(x[i]);
        }

        auto report = [&](const char* name, float ms) {
            double max_abs = 0.0, max_rel = 0.0;
            for (int i = 0; i < n; ++i) {
                double err = std::fabs(y[i] - ref[i]);
                max_abs = std::fmax(max_abs, err);
                max_rel = std::fmax(max_rel, err / std::fabs(ref[i]));
            }
            std::printf("%-9s %.3f ms  max abs err %.2e  max rel err %.2e\n",
                        name, ms, max_abs, max_rel);
        };
        // #endregion

        std::printf("\n== %d terms of sin/exp/divide on %d values ==\n",
                    kTerms, n);
        float ms = time_ms([&] {
            heavy_math<false><<<blocks, threads>>>(d_x.get(), d_y.get(), n);
        });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y.data(), d_y.get(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        report("accurate", ms);

        ms = time_ms([&] {
            heavy_math<true><<<blocks, threads>>>(d_x.get(), d_y.get(), n);
        });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y.data(), d_y.get(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        report("intrinsic", ms);

        // #region ftz-run
        float tiny = 1e-39f;   // below FLT_MIN (1.18e-38): a subnormal
        float half_tiny = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_x.get(), &tiny, sizeof(float),
                              cudaMemcpyHostToDevice));
        halve<<<1, 1>>>(d_x.get(), d_y.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&half_tiny, d_y.get(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("1e-39 * 0.5 on the GPU = %g\n", half_tiny);
        // #endregion
    }
    return 0;
}
