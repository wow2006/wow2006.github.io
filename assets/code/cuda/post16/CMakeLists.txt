cmake_minimum_required(VERSION 3.24)
project(number-formats LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(number-formats main.cu)

# The same program built with --use_fast_math, to compare the two.
add_executable(number-formats-fast main.cu)
target_compile_options(number-formats-fast PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
