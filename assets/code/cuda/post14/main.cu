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

// #region handles
using Stream = std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)>;
using Event = std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)>;
using Graph = std::unique_ptr<CUgraph_st, decltype(&cudaGraphDestroy)>;
using GraphExec =
    std::unique_ptr<CUgraphExec_st, decltype(&cudaGraphExecDestroy)>;
// #endregion

// #region kernel
__global__ void step(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += a;
    }
}
// #endregion

// Every element must equal `expected`.
bool check(const float* d_x, int n, float expected) {
    std::vector<float> x(n);
    if (cudaMemcpy(x.data(), d_x, n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    for (float v : x) {
        if (v != expected) {
            return false;
        }
    }
    return true;
}

int main() {
    // #region setup
    constexpr int n = 16384;           // tiny: 64 blocks of 256 threads
    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;
    constexpr int kernels_per_iter = 20;
    constexpr int iters = 1000;
    constexpr int total = kernels_per_iter * iters;

    auto d_x = make_device_buffer(n);
    if (!d_x) {
        return 1;
    }
    float* x = d_x.get();

    cudaStream_t raw_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&raw_stream));
    Stream stream{raw_stream, &cudaStreamDestroy};
    cudaStream_t s = stream.get();

    cudaEvent_t raw_start = nullptr, raw_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&raw_start));
    Event start{raw_start, &cudaEventDestroy};
    CUDA_CHECK(cudaEventCreate(&raw_stop));
    Event stop{raw_stop, &cudaEventDestroy};

    // One "iteration" of our workload: 20 tiny kernels in a row.
    auto enqueue_iteration = [&](float a) {
        for (int k = 0; k < kernels_per_iter; ++k) {
            step<<<blocks, threads, 0, s>>>(x, n, a);
        }
    };
    // #endregion

    // #region baseline
    enqueue_iteration(1.0f);  // warm-up
    CUDA_CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), s));

    CUDA_CHECK(cudaEventRecord(start.get(), s));
    auto c0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        enqueue_iteration(1.0f);
    }
    auto c1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop.get(), s));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float stream_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&stream_ms, start.get(), stop.get()));
    double stream_cpu_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count();
    bool stream_ok = check(x, n, float(total));
    // #endregion

    // #region capture
    auto t0 = std::chrono::steady_clock::now();

    cudaGraph_t raw_graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    enqueue_iteration(1.0f);  // recorded, not executed
    CUDA_CHECK(cudaStreamEndCapture(s, &raw_graph));
    Graph graph{raw_graph, &cudaGraphDestroy};

    cudaGraphExec_t raw_exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&raw_exec, graph.get(), 0));
    GraphExec exec{raw_exec, &cudaGraphExecDestroy};

    auto t1 = std::chrono::steady_clock::now();
    double setup_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    // #endregion

    // #region replay
    CUDA_CHECK(cudaGraphLaunch(exec.get(), s));  // warm-up
    CUDA_CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), s));

    CUDA_CHECK(cudaEventRecord(start.get(), s));
    c0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaGraphLaunch(exec.get(), s));
    }
    c1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(stop.get(), s));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float graph_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&graph_ms, start.get(), stop.get()));
    double graph_cpu_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count();
    bool graph_ok = check(x, n, float(total));
    // #endregion

    // #region update
    // Same shape, new argument: capture again and patch the existing exec.
    cudaGraph_t raw_graph2 = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    enqueue_iteration(2.0f);
    CUDA_CHECK(cudaStreamEndCapture(s, &raw_graph2));
    Graph graph2{raw_graph2, &cudaGraphDestroy};

    cudaGraphExecUpdateResultInfo info{};
    CUDA_CHECK(cudaGraphExecUpdate(exec.get(), graph2.get(), &info));

    CUDA_CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), s));
    CUDA_CHECK(cudaGraphLaunch(exec.get(), s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    bool update_ok = check(x, n, 2.0f * kernels_per_iter);
    // #endregion

    // #region report
    std::printf("%d iterations x %d kernels = %d launches, %d elements each\n",
                iters, kernels_per_iter, total, n);
    std::printf("                  GPU time   per kernel   CPU submit\n");
    std::printf("stream launches : %6.2f ms  %6.2f us  %7.2f ms  %s\n",
                stream_ms, 1000.0f * stream_ms / total, stream_cpu_ms,
                stream_ok ? "PASSED" : "FAILED");
    std::printf("graph replay    : %6.2f ms  %6.2f us  %7.2f ms  %s\n",
                graph_ms, 1000.0f * graph_ms / total, graph_cpu_ms,
                graph_ok ? "PASSED" : "FAILED");
    std::printf("capture + instantiate (once): %.1f us\n", setup_us);
    std::printf("speedup: %.2fx\n", stream_ms / graph_ms);
    std::printf("graph update with a = 2: %s\n",
                update_ok ? "PASSED" : "FAILED");
    // #endregion
    return stream_ok && graph_ok && update_ok ? 0 : 1;
}
