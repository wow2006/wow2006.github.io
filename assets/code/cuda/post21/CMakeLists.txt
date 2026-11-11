cmake_minimum_required(VERSION 3.24)
project(tile-compare LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(tile-compare main.cu)
target_link_libraries(tile-compare PRIVATE CUDA::cublas)
