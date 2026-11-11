cmake_minimum_required(VERSION 3.24)
project(multi-gpu LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(multi-gpu main.cu)
