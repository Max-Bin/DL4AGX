# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (许可信息) ...

cmake_minimum_required(VERSION 3.16)
project(bev_deploy LANGUAGES C CXX CUDA)

find_package(CUDA 11.8 REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CMake: CUDA Toolkit 11.8 not found, but is REQUIRED.")
endif()
message(STATUS "CMake: Found CUDA Version: ${CUDA_VERSION}")
message(STATUS "CMake: CUDA_INCLUDE_DIRS detected by find_package(CUDA): [${CUDA_INCLUDE_DIRS}]")

set(TARGET "x86_64") # 强制 TARGET 为 "x86_64"
message(STATUS "CMake: TARGET has been forcibly set to: [${TARGET}]")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O0 -g")
set(CMAKE_CXX_FLAGS "-fPIC -O0 -g") # 保持你原有的 CXX 标志
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}") # 通常为空，由 NVCC FLAGS 控制
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr --compiler-options -fPIC -O2 -Xptxas=-v")

# --- 关键修改：直接将 CUDA include 路径添加到 CMAKE_CXX_FLAGS ---
set(FORCED_CUDA_INCLUDE_FOR_CXX "/home/binwang/miniforge3/envs/vad/include")
if(EXISTS "${FORCED_CUDA_INCLUDE_FOR_CXX}/cuda_runtime_api.h")
    message(STATUS "CMake: Forcing CUDA include for C++: -I${FORCED_CUDA_INCLUDE_FOR_CXX}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${FORCED_CUDA_INCLUDE_FOR_CXX}")
else()
    message(WARNING "CMake: Path to force C++ CUDA include [${FORCED_CUDA_INCLUDE_FOR_CXX}] or cuda_runtime_api.h within it not found. CMAKE_CXX_FLAGS not modified with this path.")
endif()
# --- 修改结束 ---

if(${TARGET} STREQUAL "aarch64")
    message(STATUS "CMake: Configuring for aarch64 target.")
    set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(ARCH 87)
    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu) # aarch64 的 TRT include
    # 保持你原有的 aarch64 include_directories 方式
    include_directories(${TENSORRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}) 
elseif(${TARGET} STREQUAL "x86_64")
    message(STATUS "CMake: Configuring for x86_64 target.")
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT_PATH "$ENV{TRT_ROOT}")
    else()
        message(FATAL_ERROR "TRT_ROOT environment variable is not set for x86_64 target.")
    endif()
    message(STATUS "CMake: Using TRT_ROOT_PATH: [${TRT_ROOT_PATH}]")
    set(ARCH 86)
    message(STATUS "CMake: ARCH set to [${ARCH}] for x86_64.")
    set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT_PATH}/include/) # x86_64 的 TRT include
    set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT_PATH}/lib/)
    set(DEPS "")
    link_directories(${TENSORRT_LIBRARY_DIRS}) # 保持你原有的 link_directories
else()
    message(WARNING "CMake: TARGET is '[${TARGET}]', which is not 'x86_64' or 'aarch64'. ARCH will not be set appropriately!")
    set(ARCH "INVALID_TARGET_SPECIFIED") 
endif()

message(STATUS "CMake: Final ARCH value before setting target properties: [${ARCH}]")

# 全局的 include 目录 (来自你原始文件结构)
include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/kernel 
)

set(plugins_cu_srcs 
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotateKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad_kernel.cu
)

add_library(plugins_cu STATIC ${plugins_cu_srcs})

# plugins_cu 的包含目录 (来自你原始文件结构)
target_include_directories(plugins_cu PRIVATE
    ${TENSORRT_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/kernel
)
set_target_properties(plugins_cu PROPERTIES CUDA_ARCHITECTURES "${ARCH}")
set_target_properties(plugins_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(plugins_cu PROPERTIES POSITION_INDEPENDENT_CODE ON) # 保持 PIC 设置

add_library(plugins SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotate_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad.cpp
)

# plugins 的包含目录
# 由于我们已经通过 CMAKE_CXX_FLAGS 强制添加了 CUDA include 路径,
# 这里不再需要为 CUDA 路径单独添加条目。
target_include_directories(plugins PRIVATE
    ${TENSORRT_INCLUDE_DIRS}
    # CUDA include 路径现在应通过 CMAKE_CXX_FLAGS 提供
    ${CMAKE_SOURCE_DIR}/include # 项目自身的 include
    ${CMAKE_SOURCE_DIR}/kernel  # 项目自身的 kernel
)

target_link_libraries(plugins PRIVATE 
    plugins_cu 
    nvinfer    
    ${DEPS}    
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES} # 或直接 cublas，如果这个变量未定义
)