cmake_minimum_required(VERSION 3.24)
project(sgemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm main.cu)
target_link_libraries(sgemm PRIVATE CUDA::cublas)
