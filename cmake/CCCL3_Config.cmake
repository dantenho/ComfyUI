# CCCL 3.x CMake Configuration Helper
# This file helps configure projects to use CUDA C++ Core Libraries (CCCL) 3.0/3.1

cmake_minimum_required(VERSION 3.18)

# CCCL 3.x requires C++17 or newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 13.x requirement
find_package(CUDAToolkit 13.0 REQUIRED)

# CCCL 3.x: Headers moved from ${CTK_ROOT}/include/ to ${CTK_ROOT}/include/cccl/
# Use CCCL::CCCL target for proper configuration
find_package(CCCL REQUIRED)

# Example target configuration
# Uncomment and modify for your project:
#
# add_library(your_target STATIC
#     your_source.cu
#     your_source.cpp
# )
#
# target_link_libraries(your_target PRIVATE
#     CCCL::CCCL
#     CUDA::cudart
#     CUDA::cublas
#     CUDA::cusparse
# )
#
# target_compile_features(your_target PRIVATE
#     cxx_std_17
#     cuda_std_17
# )

# CUDA 13.x: Minimum compute capability 7.5 (Turing and newer)
# Blackwell support: compute capability 10.x and 12.x for CUDA Tile features
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")  # Turing, Ampere, Ada, Hopper, Blackwell

# Optional: Add Blackwell support if available
# Uncomment for RTX 50xx series support
# list(APPEND CMAKE_CUDA_ARCHITECTURES "100;120")

# CUDA 13.x: Use modern memory pool configuration
# Set via environment variable in your application:
# export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,memory_pool:device"

message(STATUS "CCCL 3.x Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
