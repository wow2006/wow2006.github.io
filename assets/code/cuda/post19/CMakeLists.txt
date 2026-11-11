cmake_minimum_required(VERSION 3.24)
project(flash-attention LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(flash-attention main.cu)
