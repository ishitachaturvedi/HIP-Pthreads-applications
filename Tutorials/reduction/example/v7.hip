// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Configuration constants
#define HIP_TEMPLATE_KERNEL_LAUNCH

// Reduction
#include <hip_utils.hpp> // HIP_CHECK
#include <tmp_utils.hpp> // tmp::static_for

// HIP API
#include <hip/hip_runtime.h>

// STL
#include <algorithm>
#include <cstddef>
#include <execution>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <utility>
#include <vector>

template<uint32_t BlockSize, uint32_t WarpSize, typename T, typename F>
__global__ __launch_bounds__(BlockSize) void kernel(
    T* front, T* back, F op, T zero_elem, uint32_t front_size)
{
    __shared__ T shared[BlockSize];

    // Overindex-safe read of input
    auto read_global_safe = [&](const uint32_t i) { return i < front_size ? front[i] : zero_elem; };

    const uint32_t tid = threadIdx.x, bid = blockIdx.x, gid = bid * (blockDim.x * 2) + tid;

    // Read input from front buffer to shared
    shared[tid] = op(read_global_safe(gid), read_global_safe(gid + blockDim.x));
    __syncthreads();

    // Shared reduction
    tmp::static_for<BlockSize / 2, tmp::greater_than<WarpSize>, tmp::divide<2>>(
        [&]<int I>()
        {
            if(tid < I)
                shared[tid] = op(shared[tid], shared[tid + I]);
            __syncthreads();
        });
    // Warp reduction
    if(tid < WarpSize)
    {
        T res = op(shared[tid], shared[tid + WarpSize]);
        tmp::static_for<WarpSize / 2, tmp::not_equal<0>, tmp::divide<2>>(
            [&]<int Delta>() { res = op(res, __shfl_down(res, Delta)); });

        // Write result from shared to back buffer
        if(tid == 0)
            back[bid] = res;
    }
}

int main(int argc, char** argv)
{
    // Device selection
    const int dev_id = argc > 1 ? std::atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev_id));

    // Query device properties
    auto [name, block_size, warp_size] = [dev_id]()
    {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, dev_id));

        return std::make_tuple(std::string{prop.name}, prop.maxThreadsPerBlock, prop.warpSize);
    }();

    std::cout << "Selected device: " << name << std::endl;
    std::cout << "Max threads per block: " << block_size << std::endl;
    std::cout << "Warp size: " << warp_size << std::endl;

    // User defined input
    constexpr std::size_t input_count = 100'000'000;
    auto kernel_op = [] __device__(unsigned lhs, unsigned rhs) { return max(lhs, rhs); };
    constexpr unsigned zero_elem = std::numeric_limits<unsigned>::min();
    std::size_t        factor    = block_size * 2;
    auto               new_size  = [factor](const std::size_t actual)
    {
        // Every pass reduces input length by 'factor'. If actual size is not divisible by factor,
        // an extra output element is produced using some number of zero_elem inputs.
        return actual / factor + (actual % factor == 0 ? 0 : 1);
    };

    // Initialize host-side storage
    std::vector<unsigned> input(input_count);
    std::iota(input.begin(), input.end(), 0);

    // Initialize device-side storage
    unsigned *front, *back;
    HIP_CHECK(hipMalloc((void**)&front, sizeof(unsigned) * input_count));
    HIP_CHECK(hipMalloc((void**)&back, new_size(input_count) * sizeof(unsigned)));

    HIP_CHECK(
        hipMemcpy(front, input.data(), input.size() * sizeof(unsigned), hipMemcpyHostToDevice));

    // Device-side reduction
    for(uint32_t curr = input_count; curr > 1;)
    {
        tmp::static_switch<std::array{32, 64, 128, 256, 512, 1024, 2048}>(
            block_size,
            [&]<int BlockSize>() noexcept
            {
                tmp::static_switch<std::array{32, 64}>(warp_size,
                                                       [&]<int WarpSize>() noexcept
                                                       {
                                                           hipLaunchKernelGGL(
                                                               kernel<BlockSize, WarpSize>,
                                                               dim3(new_size(curr)),
                                                               dim3(block_size),
                                                               0,
                                                               hipStreamDefault,
                                                               front,
                                                               back,
                                                               kernel_op,
                                                               zero_elem,
                                                               curr);
                                                       });
            });
        hip::check(hipGetLastError(), "hipKernelLaunchGGL");

        curr = new_size(curr);
        if(curr > 1)
            std::swap(front, back);
    }

    // Fetch result
    unsigned result;
    HIP_CHECK(hipMemcpy(&result, back, sizeof(unsigned), hipMemcpyDeviceToHost));

    // Free device-side resources
    HIP_CHECK(hipFree(front));
    HIP_CHECK(hipFree(back));

    if(unsigned reference = input_count - 1; result != reference)
    {
        std::cerr << "Validation FAILED!"
                  << "\n\tDevice result: " << result << "\n\tReference: " << reference << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Validation passed!" << std::endl;
        return 0;
    }
}
