cmake_minimum_required(VERSION 3.24)
project(cuda-graphs LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(cuda-graphs main.cu)
