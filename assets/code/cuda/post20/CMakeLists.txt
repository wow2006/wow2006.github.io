cmake_minimum_required(VERSION 3.24)
project(hopper-concepts LANGUAGES CXX CUDA)

# Clusters and TMA need Hopper (sm_90) or newer. "native" would pick sm_89
# on an RTX 4090, where this code does not compile.
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit REQUIRED)

add_executable(hopper-concepts main.cu)
target_link_libraries(hopper-concepts PRIVATE CUDA::cuda_driver)
