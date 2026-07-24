cmake_minimum_required(VERSION 3.13)

project(ckks_seal_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SEAL 4.1 REQUIRED)

add_executable(ckks_seal_bench ckks_seal_bench.cpp)
target_link_libraries(ckks_seal_bench PRIVATE SEAL::seal)
