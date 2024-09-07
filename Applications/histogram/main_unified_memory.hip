// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#define DEBUG_CHECK

/// \brief Calculates the 256-sized bin histogram for a block.
__global__ void
    histogram256_block(unsigned char* data, unsigned int* block_bins, const int items_per_thread)
{
    const int thread_id  = threadIdx.x;
    const int block_id   = blockIdx.x;
    const int block_size = blockDim.x;
    const int bin_size   = 256;

    extern __shared__ unsigned char thread_bins[];

    const int b_bits_length = __ffs(block_size) - 3;
    const int sh_thread_id
        = (thread_id & (1 << b_bits_length) - 1) << 2 | (thread_id >> b_bits_length);

    for(int i = 0; i < bin_size; ++i)
    {
        thread_bins[i + bin_size * sh_thread_id] = 0;
    }
    __syncthreads();

    for(int i = 0; i < items_per_thread; i++)
    {
        const unsigned int value = data[(block_id * block_size + thread_id) * items_per_thread + i];
        thread_bins[value * block_size + sh_thread_id]++;
    }
    __syncthreads();

    const int bins_per_thread = bin_size / block_size;
    for(int i = 0; i < bins_per_thread; ++i)
    {
        const int bin_sh_id = i * block_size + sh_thread_id;

        unsigned int bin_acc = 0;
        for(int j = 0; j < block_size; ++j)
        {
            bin_acc += thread_bins[bin_sh_id * block_size + j];
        }

        block_bins[block_id * bin_size + bin_sh_id] = bin_acc;
    }
}

int main()
{

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <size> <items_per_thread>" << std::endl;
        return 1;
    }

    int size = std::stoi(argv[1]);

    int items_per_thread = std::stoi(argv[2]);

    // 1. Define inputs
    //const int size              = 1024 * 1024;
    //const int items_per_thread  = 1024;
    const int threads_per_block = 128;

    const int bin_size     = 256;
    const int total_blocks = (size) / (items_per_thread * threads_per_block);

    std::vector<unsigned char> h_data(size);

    // Fixed seed for reproducibility
    const unsigned int seed = 12345;

    // Randomly generate input data with a fixed seed
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<unsigned int> distribution(0, 255);
    std::generate(h_data.begin(), h_data.end(), [&]() { return distribution(generator); });

    std::vector<unsigned int> h_bins(bin_size);
    std::vector<unsigned int> h_blockBins(bin_size * total_blocks);

    // 2. Allocate memory on device.
    unsigned char* d_data;
    unsigned int*  d_blockBins;

    float      kernel_ms = 0;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipHostMalloc(&d_blockBins, sizeof(unsigned int) * bin_size * total_blocks, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&d_data, sizeof(unsigned char) * size, hipHostMallocMapped));

    std::copy(h_data.begin(), h_data.end(), d_data);

    // 3. Launch the histogram kernel
    std::cout << "Launching 'histogram256_block' with " << total_blocks << " blocks of size "
              << threads_per_block << std::endl;

    HIP_CHECK(hipEventRecord(start));

    histogram256_block<<<dim3(total_blocks),
                         dim3(threads_per_block),
                         bin_size * threads_per_block>>>(d_data, d_blockBins, items_per_thread);
    // Check for errors.
    HIP_CHECK(hipGetLastError());

    // Get kernel execution time.
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
    std::cout << "Kernel took " << kernel_ms << " milliseconds." << std::endl;

#ifdef DEBUG_CHECK

    // 4. Copy back to host and calculate final histogram bin.
    std::copy(d_blockBins, d_blockBins + bin_size * total_blocks, h_blockBins.data());

    for(int i = 0; i < total_blocks; ++i)
    {
        for(int j = 0; j < bin_size; ++j)
        {
            int count = h_blockBins[i * bin_size + j];
            h_bins[j] += count;
        }
    }
#endif
    // 5. Free device memory.
    HIP_CHECK(hipHostFree(d_blockBins));
    HIP_CHECK(hipHostFree(d_data));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));


#ifdef DEBUG_CHECK
    // 6. Verify by calculating on host.
    int                       errors = 0;
    std::vector<unsigned int> h_verify_bins(bin_size);
    for(int i = 0; i < size; ++i)
    {
        ++h_verify_bins[h_data[i]];
    }
    for(int i = 0; i < bin_size; ++i)
    {
        errors += h_bins[i] != h_verify_bins[i];
    }
    return report_validation_result(errors);
#elif
    return 0;
#endif
}
