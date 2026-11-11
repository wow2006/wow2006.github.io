cmake_minimum_required(VERSION 3.24)
project(tensor-cores LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(tensor-cores main.cu)
target_link_libraries(tensor-cores PRIVATE CUDA::cublas)
