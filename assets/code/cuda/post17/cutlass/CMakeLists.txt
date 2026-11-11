cmake_minimum_required(VERSION 3.24)
project(cutlass-gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

# CUTLASS is header-only: download it, but do not build its own CMake project.
include(FetchContent)
FetchContent_Declare(cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.8.0.tar.gz
    SOURCE_SUBDIR do-not-build)
FetchContent_MakeAvailable(cutlass)

add_executable(cutlass-gemm main.cu)
target_include_directories(cutlass-gemm PRIVATE ${cutlass_SOURCE_DIR}/include)
target_compile_options(cutlass-gemm PRIVATE --expt-relaxed-constexpr)
