// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

//#define DEBUG_CHECK

/// \brief Given an array of n elements, this kernel implements the j-th stage within the i-th
/// step of the bitonic sort, being 0 <= i < log_2(n) and 0 <= j <= i.
__global__ void bitonic_sort_kernel(unsigned int*      array,
                                    const unsigned int step,
                                    const unsigned int stage,
                                    bool               sort_increasing)
{
    // Current thread id.
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // How many pairs of elements are ordered with the same criteria (increasingly or decreasingly)
    // within each of the bitonic subsequences computed in each step. E.g. in the step 0 we have
    // 1 pair of elements in each monotonic component of the bitonic subsequences, that is, we
    // obtain bitonic sequences of length 4.
    const unsigned int same_order_block_width = 1 << step;

    // Distance between the two elements that each thread sorts.
    const unsigned int pair_distance = 1 << (step - stage);

    // Total number of elements of each subsequence processed.
    const unsigned int sorted_block_width = 2 * pair_distance;

    // Compute indexes of the elements of the array that the thread will sort.
    const unsigned int left_id
        = (thread_id % pair_distance) + (thread_id / pair_distance) * sorted_block_width;
    const unsigned int right_id = left_id + pair_distance;

    // Get the elements of the array that the thread will sort.
    const unsigned int left_element  = array[left_id];
    const unsigned int right_element = array[right_id];

    // If the current thread is the first one ordering an element from the right component of the
    // bitonic sequence that it's computing, then the ordering criteria changes.
    if((thread_id / same_order_block_width) % 2 == 1)
        sort_increasing = !sort_increasing;

    // Compare elements and switch them if necessary.
    const unsigned int greater = (left_element > right_element) ? left_element : right_element;
    const unsigned int lesser  = (left_element > right_element) ? right_element : left_element;
    array[left_id]             = (sort_increasing) ? lesser : greater;
    array[right_id]            = (sort_increasing) ? greater : lesser;
}

/// \brief Swaps two elements if the first is greater than the second.
void swap_if_first_greater(unsigned int* a, unsigned int* b)
{
    if(*a > *b)
    {
        std::swap(*a, *b);
    }
}

/// \brief Reference CPU implementation of the bitonic sort for results verification.
void bitonic_sort_reference(unsigned int*      array,
                            const unsigned int length,
                            const bool         sort_increasing)
{
    const unsigned int half_length = length / 2;

    // For each step i' = log_2(i) - 1, 0 <= i' < log_2(length).
    for(unsigned int i = 2; i <= length; i *= 2)
    {
        // For each stage j' = log_2(i / j), 0 <= j' <= i'.
        for(unsigned int j = i; j > 1; j /= 2)
        {
            bool               increasing = sort_increasing;
            const unsigned int half_j     = j / 2;

            // Sort elements separated by distance j / 2.
            for(unsigned int k = 0; k < length; k += j)
            {
                const unsigned int k_plus_half_j = k + half_j;

                // Each time we sort i elements we must change the ordering direction.
                if((k == i) || ((i < length) && (k % i) == 0 && (k != half_length)))
                {
                    increasing = !increasing;
                }

                // Compare and sort elements.
                for(unsigned int l = k; l < k_plus_half_j; ++l)
                {
                    if(increasing)
                    {
                        swap_if_first_greater(&array[l], &array[l + half_j]);
                    }
                    else
                    {
                        swap_if_first_greater(&array[l + half_j], &array[l]);
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <log2length>" << std::endl;
        return 1; // Error exit code
    }

    // Parse the first argument as the number of steps (log2length)
    unsigned int steps = std::stoi(argv[1]);

    // Parse the second argument as the sort order (inc/dec)
    std::string sort = "inc";

    // Validate the sort order
    if(sort != "inc" && sort != "dec")
    {
        std::cerr << "The ordering must be 'dec' or 'inc', the default ordering is 'inc'." << std::endl;
        return 1; // Error exit code
    }

    const bool sort_increasing = (sort == "inc");

    // Compute length of the array to be sorted.
    const unsigned int length = 1u << steps;

    // Allocate and init random host input array. Use hipHostMalloc for pinned memory.
    unsigned int* h_array = nullptr;
    HIP_CHECK(hipHostMalloc(&h_array, length * sizeof(unsigned int), hipHostMallocDefault));

    // Generate the array
    std::mt19937 rng(10);
    std::uniform_int_distribution<unsigned int> dist(0, 99);  // Uniform distribution in range [0, 99]
    for(unsigned int i = 0; i < length; ++i)
    {
        h_array[i] = dist(rng);
    }

#ifdef DEBUG_CHECK
    std::vector<unsigned int> expected_array(h_array, h_array + length);
#endif

    std::cout << "Sorting an array of " << length << " elements using the bitonic sort."
              << std::endl;

    // Declare and allocate device memory and copy input data.
    unsigned int* d_array{};
    HIP_CHECK(hipMalloc(&d_array, length * sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(d_array, h_array, length * sizeof(unsigned int), hipMemcpyHostToDevice));

    // Number of threads in each kernel block and number of blocks in the grid.
    const unsigned int local_threads  = (length > 256) ? 256 : length / 2;
    const unsigned int global_threads = length / 2;
    const dim3         block_dim(local_threads);
    const dim3         grid_dim(global_threads / local_threads);

    // Create events to measure the execution time of the kernels.
    float      total_kernels{};
    float      kernel_ms{};
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Bitonic sort GPU algorithm: launch bitonic sort kernel for each stage of each step.
    for(unsigned int i = 0; i < steps; ++i)
    {
        // For each step i we need i + 1 stages.
        for(unsigned int j = 0; j <= i; ++j)
        {
            // Record the start event.
            HIP_CHECK(hipEventRecord(start, hipStreamDefault));

            // Launch the bitonic sort kernel on the default stream.
            bitonic_sort_kernel<<<grid_dim, block_dim, 0 /*shared memory*/, hipStreamDefault>>>(
                d_array,
                i,
                j,
                sort_increasing);

            // Check if the kernel launch was successful.
            HIP_CHECK(hipGetLastError());

            // Record the stop event and wait until the kernel execution finishes.
            HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
            HIP_CHECK(hipEventSynchronize(stop));

            // Get the execution time of the kernel and add it to the total count.
            HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
            total_kernels += kernel_ms;
        }
    }

    // Copy the sorted data back to the host.
    HIP_CHECK(hipMemcpy(h_array, d_array, length * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_array));

    // Free events variables and pinned host memory.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipHostFree(h_array));

    // Report execution time.
    std::cout << "GPU bitonic sort took " << total_kernels << " milliseconds to complete."
              << std::endl;

#ifdef DEBUG_CHECK
    // Execute CPU algorithm.
    bitonic_sort_reference(expected_array.data(), length, sort_increasing);

    // Verify results and report to user.
    unsigned int errors{};
    std::cout << "Validating results with CPU implementation." << std::endl;
    for(unsigned int i = 0; i < length; ++i)
    {
        errors += (h_array[i] != expected_array[i]);
    }
    std::cout << "Validation completed with " << errors << " errors." << std::endl;
#endif

    return 0;
}
