# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(example_name applications_monte_carlo_pi)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    message(
        FATAL_ERROR
        "Only the following values are accepted for GPU_RUNTIME: ${GPU_RUNTIMES}"
    )
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(WIN32)
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(hipcub REQUIRED)
find_package(hiprand REQUIRED)
# Workaround for hipRAND, requires manual linking with backend.
if(GPU_RUNTIME STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
else()
    find_package(rocrand REQUIRED)
endif()

add_executable(${example_name} main.hip)
add_test(NAME ${example_name} COMMAND ${example_name})

target_link_libraries(${example_name} PRIVATE hip::hipcub hip::hiprand)
# Workaround for hipRAND, requires manual linking with backend.
if(GPU_RUNTIME STREQUAL "CUDA")
    target_link_libraries(${example_name} PRIVATE CUDA::curand)
else()
    target_link_libraries(${example_name} PRIVATE roc::rocrand)
endif()
target_include_directories(${example_name} PRIVATE "../../Common")
# Workaround for rocRAND since it relies on definition "WIN32".
if(WIN32)
    target_compile_definitions(${example_name} PRIVATE WIN32)
endif()
set_source_files_properties(main.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})

install(TARGETS ${example_name})
