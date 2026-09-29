cmake_minimum_required(VERSION 3.24)
project(hello-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(hello-cuda main.cu)
