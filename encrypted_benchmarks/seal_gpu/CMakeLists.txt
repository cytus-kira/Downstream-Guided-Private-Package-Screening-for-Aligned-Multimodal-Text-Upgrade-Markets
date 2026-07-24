cmake_minimum_required(VERSION 3.18)
project(secguardfuse_seal_gpu_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(SEAL CONFIG REQUIRED)

add_executable(secguardfuse_seal_gpu_test main.cpp)
target_link_libraries(secguardfuse_seal_gpu_test PRIVATE SEAL::seal CUDA::cudart)
