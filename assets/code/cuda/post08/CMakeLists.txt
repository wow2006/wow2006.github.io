cmake_minimum_required(VERSION 3.24)
project(occupancy LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(occupancy main.cu)
