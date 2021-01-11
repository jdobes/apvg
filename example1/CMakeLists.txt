cmake_minimum_required(VERSION 2.8)

project(example1)

find_package(CUDA REQUIRED)

set(SOURCES main.cu)

list(APPEND CMAKE_CXX_FLAGS "-O2 -ffast-math -Wall")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math --ptxas-options -dlcm=cg")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} /usr/local/cuda/lib64/libcudart.so)

