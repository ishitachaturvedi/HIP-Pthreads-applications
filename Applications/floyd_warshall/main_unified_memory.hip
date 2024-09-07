// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#define DEBUG_CHECK

/// \brief Implements the k-th (0 <= k < nodes) step of Floyd-Warshall algorithm.
__global__ void floyd_warshall_kernel(unsigned int*      part_adjacency_matrix,
                                      unsigned int*      part_next_matrix,
                                      const unsigned int nodes,
                                      const unsigned int k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int d_x_y   = part_adjacency_matrix[y * nodes + x];
    int d_x_k_y = part_adjacency_matrix[y * nodes + k] + part_adjacency_matrix[k * nodes + x];

    if(d_x_k_y < d_x_y)
    {
        part_adjacency_matrix[y * nodes + x] = d_x_k_y;
        part_next_matrix[y * nodes + x]      = k;
    }
}

/// \brief Reference CPU implementation of Floyd-Warshall algorithm for results verification.
void floyd_warshall_reference(unsigned int*      adjacency_matrix,
                              unsigned int*      next_matrix,
                              const unsigned int nodes)
{
    for(unsigned int k = 0; k < nodes; k++)
    {
        for(unsigned int x = 0; x < nodes; x++)
        {
            const unsigned int row_x = x * nodes;
            for(unsigned int y = 0; y < nodes; y++)
            {
                const unsigned int d_x_y = adjacency_matrix[row_x + y];
                const unsigned int d_x_k = adjacency_matrix[row_x + k];
                const unsigned int d_k_y = adjacency_matrix[k * nodes + y];
                const unsigned int d_x_k_y = d_x_k + d_k_y;

                if(d_x_k_y < d_x_y)
                {
                    adjacency_matrix[row_x + y] = d_x_k_y;
                    next_matrix[row_x + y]      = k;
                }
            }
        }
    }
}

/// \brief Adds to a command line parser the necessary options for this example.
template<unsigned int BlockSize>
void configure_parser(cli::Parser& parser)
{
    constexpr unsigned int nodes      = 16;
    constexpr unsigned int iterations = 1;

    static_assert(((nodes % BlockSize == 0)),
                  "Number of nodes must be a positive multiple of BlockSize");
    static_assert(((iterations > 0)), "Number of iterations must be at least 1");

    parser.set_optional<unsigned int>("n", "nodes", nodes, "Number of nodes in the graph.");
    parser.set_optional<unsigned int>("i",
                                      "iterations",
                                      iterations,
                                      "Number of times the algorithm is executed.");
}

int main(int argc, char* argv[])
{
    constexpr unsigned int block_size = 16;

    cli::Parser parser(argc, argv);
    configure_parser<block_size>(parser);
    parser.run_and_exit_if_error();

    const unsigned int nodes      = parser.get<unsigned int>("n");
    const unsigned int iterations = parser.get<unsigned int>("i");

    if(nodes % block_size)
    {
        std::cout << "Number of nodes must be a positive multiple of block_size ("
                  << std::to_string(block_size) << ")." << std::endl;
        return -1;
    }
    if(iterations == 0)
    {
        std::cout << "Number of iterations must be at least 1." << std::endl;
        return -1;
    }

    const unsigned int size       = nodes * nodes;
    const unsigned int size_bytes = size * sizeof(unsigned int);

    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(nodes / block_size, nodes / block_size);

    std::vector<unsigned int> adjacency_matrix(size);
    std::iota(adjacency_matrix.begin(), adjacency_matrix.end(), 1);
    for(unsigned int x = 0; x < nodes; x++)
    {
        adjacency_matrix[x * nodes + x] = 0;
    }

    std::vector<unsigned int> next_matrix(size);
    for(unsigned int x = 0; x < nodes; x++)
    {
        for(unsigned int y = 0; y < x; y++)
        {
            next_matrix[x * nodes + y] = x;
            next_matrix[y * nodes + x] = y;
        }
        next_matrix[x * nodes + x] = x;
    }

    std::vector<unsigned int> expected_adjacency_matrix(adjacency_matrix);
    std::vector<unsigned int> expected_next_matrix(next_matrix);

    unsigned int* part_adjacency_matrix = nullptr;
    unsigned int* part_next_matrix      = nullptr;
    unsigned int* d_adjacency_matrix    = nullptr;
    unsigned int* d_next_matrix         = nullptr;

    double kernel_time = 0;

    std::cout << "Executing Floyd-Warshall algorithm for " << iterations
              << " iterations with a complete graph of " << nodes << " nodes." << std::endl;

    HIP_CHECK(hipHostMalloc(&part_adjacency_matrix, size_bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&part_next_matrix, size_bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&d_adjacency_matrix, size_bytes, hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(&d_next_matrix, size_bytes, hipHostMallocMapped));

    std::copy(adjacency_matrix.begin(), adjacency_matrix.end(), part_adjacency_matrix);
    std::copy(next_matrix.begin(), next_matrix.end(), part_next_matrix);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(unsigned int i = 0; i < iterations; ++i)
    {
        std::copy(part_adjacency_matrix, part_adjacency_matrix + size, d_adjacency_matrix);
        std::copy(part_next_matrix, part_next_matrix + size, d_next_matrix);

        float kernel_ms{};

        for(unsigned int k = 0; k < nodes; ++k)
        {
            HIP_CHECK(hipEventRecord(start, hipStreamDefault));

            floyd_warshall_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
                d_adjacency_matrix, d_next_matrix, nodes, k);

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
            HIP_CHECK(hipEventSynchronize(stop));
            HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
            kernel_time += kernel_ms;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

#ifdef DEBUG_CHECK
    std::copy(d_adjacency_matrix, d_adjacency_matrix + size, adjacency_matrix.data());
    std::copy(d_next_matrix, d_next_matrix + size, next_matrix.data());
#endif

    HIP_CHECK(hipHostFree(part_adjacency_matrix));
    HIP_CHECK(hipHostFree(part_next_matrix));
    HIP_CHECK(hipHostFree(d_adjacency_matrix));
    HIP_CHECK(hipHostFree(d_next_matrix));

#ifdef DEBUG_CHECK
    kernel_time /= iterations;
    std::cout << "The mean time needed for each iteration has been " << kernel_time << "ms."
              << std::endl;

    floyd_warshall_reference(expected_adjacency_matrix.data(), expected_next_matrix.data(), nodes);

    unsigned int errors = 0;
    std::cout << "Validating results with CPU implementation." << std::endl;
    for(unsigned int i = 0; i < size; ++i)
    {
        errors += (adjacency_matrix[i] - expected_adjacency_matrix[i] != 0);
        errors += (next_matrix[i] - expected_next_matrix[i] != 0);
    }

    if(errors)
    {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }
#endif
    return 0;
}
