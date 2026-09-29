cmake_minimum_required(VERSION 3.24)
project(vector-add LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(vector-add main.cu)
