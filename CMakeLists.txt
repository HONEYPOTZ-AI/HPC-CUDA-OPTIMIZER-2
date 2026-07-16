cmake_minimum_required(VERSION 3.22)
project(hpc_cuda_optimizer LANGUAGES CXX CUDA)

# ============================================================
# Toolchain & standards
# ============================================================
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# GPU architecture targets
# ============================================================
# sm_80 = A100, sm_90 = H100, sm_89 = L40/RTX 4090
set(CMAKE_CUDA_ARCHITECTURES "80;89;90" CACHE STRING "CUDA architectures")

# ============================================================
# Dependencies
# ============================================================
find_package(CUDAToolkit REQUIRED)
find_package(MPI         REQUIRED)

# NCCL — locate via pkg-config or manual paths
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS $ENV{NCCL_HOME}/include
        /usr/include
        /usr/local/cuda/include)
find_library(NCCL_LIBRARY nccl
  HINTS $ENV{NCCL_HOME}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR
    "NCCL not found. Set NCCL_HOME or specify -DNCCL_INCLUDE_DIR / -DNCCL_LIBRARY")
endif()
message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")

# ============================================================
# Compiler flags
# ============================================================
set(CXX_WARN_FLAGS  -Wall -Wextra -Wpedantic)
set(CUDA_ARCH_FLAGS --generate-code=arch=compute_80,code=sm_80
                    --generate-code=arch=compute_89,code=sm_89
                    --generate-code=arch=compute_90,code=sm_90)

# Release CUDA flags: fast-math, line-info for profiling
set(CUDA_RELEASE_FLAGS
    -O3
    --use_fast_math
    -lineinfo
    -Xcompiler=-O3,-march=native)

# ============================================================
# Common interface library (propagates includes & libs)
# ============================================================
add_library(common_deps INTERFACE)
target_include_directories(common_deps INTERFACE
  ${NCCL_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(common_deps INTERFACE
  ${NCCL_LIBRARY}
  MPI::MPI_CXX
  CUDA::cudart
  CUDA::cuda_driver)
target_compile_options(common_deps INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARN_FLAGS}>)

# ============================================================
# Kernel object library (compiled once, linked into binaries)
# ============================================================
add_library(optimizer_kernels OBJECT
  src/optimizer_kernels.cu)
target_link_libraries(optimizer_kernels PUBLIC common_deps)
target_compile_options(optimizer_kernels PRIVATE
  $<$<BUILD_TYPE:Release>:${CUDA_RELEASE_FLAGS}>
  $<$<BUILD_TYPE:Debug>:-G -g>)
set_target_properties(optimizer_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Executables
# ============================================================

# 1. Main optimizer binary
add_executable(optimizer_main src/optimizer_main.cpp)
target_link_libraries(optimizer_main PRIVATE common_deps optimizer_kernels)

# 2. Reduction micro-benchmark (CUDA only)
add_executable(bench_reduce bench/bench_reduce.cu)
target_link_libraries(bench_reduce PRIVATE common_deps)
target_compile_options(bench_reduce PRIVATE
  $<$<BUILD_TYPE:Release>:${CUDA_RELEASE_FLAGS}>)

# 3. ZeRO-3 MPI scatter/gather benchmark
add_executable(bench_zero3 bench/bench_zero3.cpp)
target_link_libraries(bench_zero3 PRIVATE common_deps)

# 4. ZeRO-3 NCCL single-node AllReduce benchmark
add_executable(bench_zero3_nccl bench/bench_zero3_nccl.cpp)
target_link_libraries(bench_zero3_nccl PRIVATE common_deps)

# 5. ZeRO-3 NCCL multi-node AllReduce benchmark
add_executable(bench_zero3_nccl_multi bench/bench_zero3_nccl_multi.cpp)
target_link_libraries(bench_zero3_nccl_multi PRIVATE common_deps)

# ============================================================
# Install targets
# ============================================================
install(TARGETS
    optimizer_main bench_reduce bench_zero3 bench_zero3_nccl bench_zero3_nccl_multi
  RUNTIME DESTINATION bin)
install(DIRECTORY scripts/ DESTINATION scripts
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(DIRECTORY slurm/ DESTINATION slurm
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# ============================================================
# Tests (CTest integration)
# ============================================================
enable_testing()

# Quick smoke-test: single-step AdamW on current machine (needs ≥1 GPU)
add_test(NAME smoke_adamw
  COMMAND optimizer_main --optimizer adamw --params 1048576 --steps 5 --verbose)

add_test(NAME smoke_sgd
  COMMAND optimizer_main --optimizer sgd   --params 1048576 --steps 5 --verbose)

add_test(NAME smoke_lamb
  COMMAND optimizer_main --optimizer lamb  --params 1048576 --steps 5 --verbose)

add_test(NAME reduce_bench
  COMMAND bench_reduce)

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "=== HPC CUDA Optimizer Build Config ===")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  MPI               : ${MPI_CXX_COMPILER}")
message(STATUS "  NCCL              : ${NCCL_LIBRARY}")
message(STATUS "  Install prefix    : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
