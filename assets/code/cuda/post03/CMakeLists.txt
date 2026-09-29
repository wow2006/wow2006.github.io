cmake_minimum_required(VERSION 3.24)
project(measure-speed LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(measure-speed main.cu)
