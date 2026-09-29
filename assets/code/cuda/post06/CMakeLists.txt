cmake_minimum_required(VERSION 3.24)
project(transpose LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(transpose main.cu)
