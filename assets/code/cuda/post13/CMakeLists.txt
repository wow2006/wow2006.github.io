cmake_minimum_required(VERSION 3.24)
project(unified-memory LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(unified-memory main.cu)
