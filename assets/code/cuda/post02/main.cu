#include <cstdio>
#include <cstdlib>
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

constexpr int radius = 2;  // 5x5 box blur

// #region pixel
// Average of the (2 * radius + 1)^2 neighbours of (x, y) that are inside the image.
__host__ __device__ float blur_pixel(const float* in, int width, int height,
                                     int x, int y) {
    float sum = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += in[ny * width + nx];
                ++count;
            }
        }
    }
    return sum / count;
}
// #endregion

// #region blur2d
__global__ void blur2d(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[y * width + x] = blur_pixel(in, width, height, x, y);
    }
}
// #endregion

// #region stride
__global__ void blur_grid_stride(const float* in, float* out,
                                 int width, int height) {
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;
         y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;
             x += blockDim.x * gridDim.x) {
            out[y * width + x] = blur_pixel(in, width, height, x, y);
        }
    }
}
// #endregion

// #region diverge
// Threads take path A or B depending on (threadIdx.x / group) % 2.
// group = 1:  neighbours in a warp disagree  -> the warp runs both paths.
// group = 32: every thread in a warp agrees  -> each warp runs one path.
__global__ void branchy(float* out, int n, int group) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = i * 1e-6f;
    if ((threadIdx.x / group) % 2 == 0) {
        for (int k = 0; k < 256; ++k) {
            v = sinf(v) + 1.0f;
        }
    } else {
        for (int k = 0; k < 256; ++k) {
            v = cosf(v) - 1.0f;
        }
    }
    out[i] = v;
}
// #endregion

// #region timer
// Average time of one launch, in milliseconds. Post 03 explains this properly.
template <typename Launch>
float time_ms(Launch launch, int reps = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    launch();  // warm-up
    cudaEventRecord(start);
    for (int r = 0; r < reps; ++r) {
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

int main(int argc, char** argv) {
    // #region image
    constexpr int width = 4000;
    constexpr int height = 3000;
    constexpr int pixels = width * height;
    constexpr std::size_t bytes = pixels * sizeof(float);

    // A synthetic grayscale image: values 0..255, no file needed.
    std::vector<float> image(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image[y * width + x] = static_cast<float>((x ^ y) & 0xFF);
        }
    }
    // #endregion

    // #region reference
    // CPU reference result
    std::vector<float> expected(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            expected[y * width + x] = blur_pixel(image.data(), width, height, x, y);
        }
    }
    // #endregion

    auto d_in = make_device_buffer(pixels);
    auto d_out = make_device_buffer(pixels);
    if (!d_in || !d_out) {
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in.get(), image.data(), bytes, cudaMemcpyHostToDevice));

    std::vector<float> result(pixels);
    auto count_errors = [&] {
        int errors = 0;
        for (int i = 0; i < pixels; ++i) {
            if (result[i] != expected[i]) {
                ++errors;
            }
        }
        return errors;
    };

    std::printf("Image %d x %d, %dx%d box blur\n", width, height,
                2 * radius + 1, 2 * radius + 1);

    // #region launch2d
    // Block shape from the command line, e.g. ./blur 32 8. Default 16x16.
    int bx = argc > 2 ? std::atoi(argv[1]) : 16;
    int by = argc > 2 ? std::atoi(argv[2]) : 16;
    dim3 block(bx, by);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    blur2d<<<grid, block>>>(d_in.get(), d_out.get(), width, height);
    CUDA_CHECK(cudaGetLastError());
    // #endregion
    CUDA_CHECK(cudaMemcpy(result.data(), d_out.get(), bytes, cudaMemcpyDeviceToHost));
    float ms = time_ms([&] {
        blur2d<<<grid, block>>>(d_in.get(), d_out.get(), width, height);
    });
    std::printf("blur2d:           grid %4u x %-4u block %2u x %-2u  %.3f ms  %s\n",
                grid.x, grid.y, block.x, block.y, ms,
                count_errors() == 0 ? "PASSED" : "FAILED");

    // #region launchstride
    CUDA_CHECK(cudaMemset(d_out.get(), 0, bytes));
    dim3 small_grid(8, 8);
    blur_grid_stride<<<small_grid, block>>>(d_in.get(), d_out.get(), width, height);
    CUDA_CHECK(cudaGetLastError());
    // #endregion
    CUDA_CHECK(cudaMemcpy(result.data(), d_out.get(), bytes, cudaMemcpyDeviceToHost));
    ms = time_ms([&] {
        blur_grid_stride<<<small_grid, block>>>(d_in.get(), d_out.get(), width, height);
    });
    std::printf("blur_grid_stride: grid %4u x %-4u block %2u x %-2u  %.3f ms  %s\n",
                small_grid.x, small_grid.y, block.x, block.y, ms,
                count_errors() == 0 ? "PASSED" : "FAILED");

    // #region launchdiverge
    constexpr int threads = 256;
    constexpr int blocks = (pixels + threads - 1) / threads;
    for (int group : {32, 1}) {
        float t = time_ms([&] {
            branchy<<<blocks, threads>>>(d_out.get(), pixels, group);
        });
        CUDA_CHECK(cudaGetLastError());
        std::printf("branchy group=%-2d  %.3f ms\n", group, t);
    }
    // #endregion
    return 0;
}
