cmake_minimum_required(VERSION 3.24)
project(vector-add-nvtx LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(vector-add-nvtx main.cu)
