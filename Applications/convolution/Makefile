# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE_HIP := convolution_HIP
EXAMPLE_pthreads_coarse := convolution_pthreads_coarse_grained
EXAMPLE_pthreads_fine := convolution_pthreads_fine_grained
EXAMPLE_pthreads_fine_extra := convolution_pthreads_fine_grained_extra
COMMON_INCLUDE_DIR := ../../Common
GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

ifeq ($(GPU_RUNTIME), CUDA)
	ICXXFLAGS += -x cu
	ICPPFLAGS += -isystem $(HIP_INCLUDE_DIR)
else ifeq ($(GPU_RUNTIME), HIP)
	CXXFLAGS ?= -Wall -Wextra
else
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be either CUDA or HIP)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE_HIP): main_unified_memory.hip $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/cmdparser.hpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o bin/$@ $< $(ILDLIBS)

$(EXAMPLE_pthreads_coarse):
	g++ convolution_pthreads_work_stealing_coarse_grained.cpp -o bin/convolution_pthreads_coarse_grained -lpthread -static

$(EXAMPLE_pthreads_fine):
	g++ convolution_pthreads_work_stealing_fine_grained.cpp -o bin/convolution_pthreads_fine_grained -lpthread -static

$(EXAMPLE_pthreads_fine_extra):
	g++ convolution_pthreads_work_stealing_fine_grained_extra_code.cpp -o bin/convolution_pthreads_fine_grained_extra -lpthread -static

clean:
	$(RM) $(EXAMPLE_HIP) $(EXAMPLE_pthreads)

.PHONY: clean
