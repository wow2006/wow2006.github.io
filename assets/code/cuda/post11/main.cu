#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
// #region shim
// Workaround for -U_GNU_SOURCE (see post 00): Thrust pulls in <mutex> and
// <shared_mutex>, which call four glibc functions that are only declared
// when _GNU_SOURCE is set. Post 09 needed the first two for CUB.
#include <pthread.h>
extern "C" {
int pthread_cond_clockwait(pthread_cond_t*, pthread_mutex_t*, clockid_t,
                           const timespec*);
int pthread_mutex_clocklock(pthread_mutex_t*, clockid_t, const timespec*);
int pthread_rwlock_clockrdlock(pthread_rwlock_t*, clockid_t, const timespec*);
int pthread_rwlock_clockwrlock(pthread_rwlock_t*, clockid_t, const timespec*);
}
// #endregion
#include <cuda_runtime.h>

// #region includes
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/atomic>
#include <cuda/std/span>
#include <cublas_v2.h>
#include <cufft.h>
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

int thrust_demo() {
    // #region thrust
    constexpr int n = 1'000'000;
    constexpr float a = 2.0f;

    thrust::device_vector<float> x(n, 1.0f);  // cudaMalloc + fill
    thrust::device_vector<float> y(n, 2.0f);

    // y = a * x + y (SAXPY)
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
                      [a] __host__ __device__ (float xi, float yi) {
                          return a * xi + yi;
                      });

    float sum = thrust::reduce(y.begin(), y.end());  // 0.0f and + by default
    float sum_sq = thrust::transform_reduce(
        y.begin(), y.end(),
        [] __host__ __device__ (float v) { return v * v; },  // transform
        0.0f, cuda::std::plus<float>{});                    // reduce
    float first = y[0];  // one element copied back to the host
    // #endregion

    std::printf("thrust: y[0] = %.1f, sum = %.1f, norm = %.1f\n",
                first, sum, std::sqrt(sum_sq));

    // #region cublas
    thrust::device_vector<float> y_blas(n, 2.0f);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    cublasStatus_t status = cublasSaxpy(handle, n, &a,
                                        thrust::raw_pointer_cast(x.data()), 1,
                                        thrust::raw_pointer_cast(y_blas.data()), 1);
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSaxpy: %s\n", cublasGetStatusString(status));
        return 1;
    }
    bool same = thrust::equal(y.begin(), y.end(), y_blas.begin());
    // #endregion
    std::printf("cublas: saxpy matches thrust: %s\n", same ? "yes" : "no");
    return same ? 0 : 1;
}

// From post 03: average GPU time of one call to `work`, in milliseconds.
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

double ms_since(std::chrono::steady_clock::time_point start) {
    using ms = std::chrono::duration<double, std::milli>;
    return ms(std::chrono::steady_clock::now() - start).count();
}

int sort_demo() {
    constexpr int n = 1 << 24;  // 16.7M keys, 64 MiB
    std::vector<unsigned> h_keys(n);
    std::mt19937 rng(42);
    for (auto& k : h_keys) {
        k = rng();
    }

    // #region std-sort
    std::vector<unsigned> h_sorted = h_keys;
    auto t0 = std::chrono::steady_clock::now();
    std::sort(h_sorted.begin(), h_sorted.end());
    double cpu_ms = ms_since(t0);
    // #endregion

    // #region thrust-sort
    thrust::device_vector<unsigned> d_keys = h_keys;  // host -> device copy
    // Sort time alone, data already on the GPU. Radix sort does the same
    // work whether the input is sorted or not, so re-sorting is fair.
    float thrust_ms = time_ms([&] { thrust::sort(d_keys.begin(), d_keys.end()); });

    // What a CPU program actually pays: copy in, sort, copy out.
    std::vector<unsigned> h_result(n);
    t0 = std::chrono::steady_clock::now();
    thrust::device_vector<unsigned> d_trip = h_keys;
    thrust::sort(d_trip.begin(), d_trip.end());
    thrust::copy(d_trip.begin(), d_trip.end(), h_result.begin());
    double round_trip_ms = ms_since(t0);
    // #endregion

    // #region cub-sort
    thrust::device_vector<unsigned> d_in = h_keys, d_out(n);
    const unsigned* in = thrust::raw_pointer_cast(d_in.data());
    unsigned* out = thrust::raw_pointer_cast(d_out.data());

    // 1st call: d_temp is nullptr, so CUB only writes the size it needs.
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, in, out, n));
    thrust::device_vector<char> temp(temp_bytes);
    d_temp = thrust::raw_pointer_cast(temp.data());

    // 2nd call: does the sort.
    float cub_ms = time_ms([&] {
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, in, out, n);
    });
    CUDA_CHECK(cudaGetLastError());
    // #endregion

    bool ok = h_result == h_sorted &&
              thrust::equal(d_out.begin(), d_out.end(), h_sorted.begin());

    std::printf("sort %d keys: %s\n", n, ok ? "all three match" : "MISMATCH");
    std::printf("  std::sort (1 CPU core):  %8.1f ms\n", cpu_ms);
    std::printf("  thrust incl. copies:     %8.1f ms\n", round_trip_ms);
    std::printf("  thrust::sort:            %8.2f ms\n", thrust_ms);
    std::printf("  cub::DeviceRadixSort:    %8.2f ms (%zu KiB temp)\n",
                cub_ms, temp_bytes / 1024);
    return ok ? 0 : 1;
}

// #region kernel
__global__ void count_above(cuda::std::span<const float> v, float threshold,
                            int* count) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v.size() && v[i] > threshold) {
        cuda::atomic_ref<int, cuda::thread_scope_device> c(*count);
        c.fetch_add(1, cuda::std::memory_order_relaxed);
    }
}
// #endregion

int libcudacxx_demo() {
    // #region libcudacxx
    constexpr int n = 1'000'000;
    constexpr float threshold = 749'999.5f;
    thrust::device_vector<float> v(n);
    thrust::sequence(v.begin(), v.end());  // 0, 1, 2, ..., n - 1
    thrust::device_vector<int> count(1, 0);

    cuda::std::span<const float> view(thrust::raw_pointer_cast(v.data()), n);
    constexpr int threads = 256;
    count_above<<<(n + threads - 1) / threads, threads>>>(
        view, threshold, thrust::raw_pointer_cast(count.data()));
    CUDA_CHECK(cudaGetLastError());

    int expected = thrust::count_if(v.begin(), v.end(),
        [=] __host__ __device__ (float x) { return x > threshold; });
    int got = count[0];
    // #endregion
    std::printf("libcu++: %d values above %.1f (count_if says %d)\n",
                got, threshold, expected);
    return got == expected ? 0 : 1;
}

int cufft_demo() {
    // #region cufft
    constexpr int n = 1024;
    constexpr int freq = 50;  // cycles over the whole signal
    const float pi = std::acos(-1.0f);

    std::vector<cufftComplex> h(n);
    for (int t = 0; t < n; ++t) {
        h[t] = {std::sin(2.0f * pi * freq * t / n), 0.0f};
    }
    thrust::device_vector<cufftComplex> signal = h;
    cufftComplex* d_signal = thrust::raw_pointer_cast(signal.data());

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        return 1;
    }
    cufftResult result = cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    cufftDestroy(plan);
    if (result != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecC2C failed: %d\n", result);
        return 1;
    }

    thrust::device_vector<float> mag(n);
    thrust::transform(signal.begin(), signal.end(), mag.begin(),
                      [] __host__ __device__ (cufftComplex z) {
                          return cuCabsf(z);
                      });
    // Real input gives a mirrored spectrum, so only look at the first half.
    auto peak = thrust::max_element(mag.begin(), mag.begin() + n / 2);
    int bin = peak - mag.begin();
    float peak_mag = *peak;
    // #endregion
    std::printf("cufft: peak at bin %d, magnitude %.1f (expected bin %d, %d)\n",
                bin, peak_mag, freq, n / 2);
    return bin == freq ? 0 : 1;
}

int main() {
    std::printf("CCCL %d.%d.%d\n", CCCL_MAJOR_VERSION, CCCL_MINOR_VERSION,
                CCCL_PATCH_VERSION);
    if (thrust_demo() != 0 || sort_demo() != 0 || libcudacxx_demo() != 0 ||
        cufft_demo() != 0) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
