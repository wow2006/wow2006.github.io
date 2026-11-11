cmake_minimum_required(VERSION 3.24)
project(scan-histogram LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(scan scan.cu)
add_executable(histogram histogram.cu)
