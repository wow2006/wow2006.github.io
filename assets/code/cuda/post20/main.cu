// Hopper+ only (sm_90 and newer). Compiles here, does not run on an RTX 4090.
#include <cstdio>
#include <memory>
#include <vector>
#include <cooperative_groups.h>
#include <cuda.h>             // CUtensorMap, cuTensorMapEncodeTiled (driver API)
#include <cuda/barrier>       // cuda::barrier + TMA wrappers
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace cde = cuda::device::experimental;

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

// #region cluster
constexpr int kClusterSize = 4;

// Each block sums its 256 elements into its own shared memory. Block 0 of
// every cluster then reads the other blocks' sums directly (DSMEM).
__global__ void __cluster_dims__(kClusterSize, 1, 1)
cluster_sum(const float* in, float* out, int n) {
    __shared__ float partial;
    cg::cluster_group cluster = cg::this_cluster();

    if (threadIdx.x == 0) partial = 0.0f;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&partial, in[i]);

    cluster.sync();  // every block's partial is now visible to the cluster

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) {
            float* remote = cluster.map_shared_rank(&partial, r);
            total += *remote;  // a load from another SM's shared memory
        }
        out[blockIdx.x / kClusterSize] = total;
    }

    cluster.sync();  // keep our shared memory alive until rank 0 is done
}
// #endregion

// #region tma-kernel
constexpr int kTile = 32;

// Each block moves one 32x32 tile: global -> shared with one TMA
// instruction, adds 1, then shared -> global with another.
__global__ void tma_add_one(const __grid_constant__ CUtensorMap map) {
    __shared__ alignas(128) float tile[kTile][kTile];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;  // an mbarrier

    int x = blockIdx.x * kTile;
    int y = blockIdx.y * kTile;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();  // make the barrier visible to TMA
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0) {
        // One thread starts the whole 4 KB copy...
        cde::cp_async_bulk_tensor_2d_global_to_shared(&tile, &map, x, y, bar);
        // ...and tells the barrier how many bytes to wait for.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));  // all threads wait until the bytes land

    for (int k = threadIdx.x; k < kTile * kTile; k += blockDim.x) {
        tile[k / kTile][k % kTile] += 1.0f;
    }

    cde::fence_proxy_async_shared_cta();  // our writes -> visible to TMA
    __syncthreads();

    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&map, x, y, &tile);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
}
// #endregion

// #region tensor-map
// Describe a rows x cols row-major float matrix and a 32x32 box to the TMA unit.
bool make_tensor_map(CUtensorMap* map, float* data, int rows, int cols) {
    cuuint64_t dims[2] = {cuuint64_t(cols), cuuint64_t(rows)};  // fastest dim first
    cuuint64_t strides[1] = {cols * sizeof(float)};              // bytes, dims 1..
    cuuint32_t box[2] = {kTile, kTile};
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, data, dims, strides, box,
        elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS;
}
// #endregion

int main() {
    // #region guard
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    if (major < 9) {
        std::printf("Compute capability %d.%d: clusters and TMA need 9.0+. "
                    "Nothing to run.\n", major, minor);
        return 0;
    }
    // #endregion

    // Part 1: cluster sum over DSMEM.
    constexpr int threads = 256;
    constexpr int blocks = 64;  // a multiple of kClusterSize
    constexpr int n = blocks * threads;
    constexpr int clusters = blocks / kClusterSize;

    std::vector<float> h_in(n, 1.0f), h_sums(clusters);
    auto d_in = make_device_buffer(n);
    auto d_sums = make_device_buffer(clusters);
    if (!d_in || !d_sums) return 1;

    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    cluster_sum<<<blocks, threads>>>(d_in.get(), d_sums.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums.get(), clusters * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int errors = 0;
    for (float s : h_sums) {
        if (s != kClusterSize * threads) ++errors;  // 4 blocks x 256 ones
    }
    std::printf("cluster_sum: %d clusters, %s\n", clusters,
                errors == 0 ? "PASSED" : "FAILED");

    // Part 2: TMA tile copy on a 256 x 256 matrix.
    constexpr int rows = 256, cols = 256;
    std::vector<float> h_m(rows * cols);
    for (int i = 0; i < rows * cols; ++i) h_m[i] = i % 1000;
    auto d_m = make_device_buffer(rows * cols);
    if (!d_m) return 1;
    CUDA_CHECK(cudaMemcpy(d_m.get(), h_m.data(), h_m.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUtensorMap map{};
    if (!make_tensor_map(&map, d_m.get(), rows, cols)) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
        return 1;
    }
    tma_add_one<<<dim3(cols / kTile, rows / kTile), 128>>>(map);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_out(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_m.get(), h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int tma_errors = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (h_out[i] != h_m[i] + 1.0f) ++tma_errors;
    }
    std::printf("tma_add_one: %s (%d errors)\n",
                tma_errors == 0 ? "PASSED" : "FAILED", tma_errors);
    return errors == 0 && tma_errors == 0 ? 0 : 1;
}
