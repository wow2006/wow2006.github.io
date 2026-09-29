cmake_minimum_required(VERSION 3.24)
project(tiled-matmul LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(tiled-matmul main.cu)
target_link_libraries(tiled-matmul PRIVATE CUDA::cublas)
