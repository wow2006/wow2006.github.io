cmake_minimum_required(VERSION 3.24)
project(cccl-libs LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(cccl-libs main.cu)
target_compile_options(cccl-libs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(cccl-libs PRIVATE CUDA::cublas CUDA::cufft)
