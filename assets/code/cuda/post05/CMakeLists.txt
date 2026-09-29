cmake_minimum_required(VERSION 3.24)
project(coalescing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(coalescing main.cu)
