#include "cmdparser.hpp"
#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ostream>
#include <random>
#include <string>
#include <vector>

//#define DEBUG_CHECK

// clang-format off
/// \brief Convolution filter using arbitrary values
const constexpr std::array<float, 5 * 5> convolution_filter_5x5 = {1.0f,  3.0f, 0.0f,  -2.0f, -0.0f, 
                                                                   1.0f,  4.0f, 0.0f,  -8.0f, -4.0f,
                                                                   2.0f,  7.0f, 0.0f, -12.0f, -0.0f,
                                                                   2.0f,  3.0f, 1.5f,  -8.0f, -4.0f,
                                                                   0.0f,  1.0f, 0.0f,  -2.0f, -0.0f};
// clang-format on

/// \brief allocate memory in constant address space for the mask on the device
__constant__ float d_mask[5 * 5];

/// \brief Implements a convolution for an input grid \p input and a \p d_mask that is defined in constant memory. The \p input needs
/// to be padded such that \p mask_size is taken into account, i.e. padded_width = floor(mask_width/2) * 2 + width
/// and padded_height = floor(mask_height/2) * 2 + height
template<size_t MaskWidth = 5>
__global__ void convolution(const float* input, float* output, const uint2 input_dimensions)
{
    const size_t x            = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y            = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t width        = input_dimensions.x;
    const size_t height       = input_dimensions.y;
    const size_t padded_width = width + (MaskWidth / 2) * 2;

    // Check if the currently computed element is inside the grid domain.
    if(x >= width || y >= height)
        return;

    // Temporary storage variables.
    float        sum              = 0.0f;
    const size_t convolution_base = y * padded_width + x;

    // Iterate over the mask in both x and y direction.
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t mask_index         = mask_index_y * MaskWidth + mask_index_x;
            const size_t convolution_offset = mask_index_y * padded_width + mask_index_x;
            sum += input[convolution_base + convolution_offset] * d_mask[mask_index];
        }
    }

    output[y * width + x] = sum;
}

template<typename T>
void print_grid(const std::vector<T>& vec, int width)
{
    size_t num_rows = vec.size() / width;
    auto   it       = vec.begin();
    for(size_t i = 0; i < num_rows; i++)
    {
        std::copy(it, it + width, std::ostream_iterator<T>(std::cout, " "));
        std::cout << std::endl;
        it += width;
    }
}

/// \brief Reference CPU implementation of convolution for results verification.
template<typename mask_type>
void convolution_reference(std::vector<float>&       verificationOutput,
                           const std::vector<float>& paddedInput,
                           const mask_type&          mask,
                           const unsigned int        height,
                           const unsigned int        width,
                           const unsigned int        mask_width)
{
    // padded_width = width + floor(mask_width / 2) * 2
    const unsigned int padded_width = width + (mask_width / 2) * 2;
    // Iterate over the provided grid.
    for(unsigned int y = 0; y < height; y++)
    {
        for(unsigned int x = 0; x < width; x++)
        {
            // temporary for summation.
            float sum = 0.0f;
            // Iterate over the mask for the given element.
            for(unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y)
            {
                for(unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x)
                {
                    unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                    unsigned int input_index
                        = (y + mask_index_y) * padded_width + (x + mask_index_x);
                    sum += paddedInput[input_index] * mask[mask_index];
                }
            }
            verificationOutput[(y * width + x)] = sum;
        }
    }
}

/// \brief Adds to a command line parser the necessary options for this example.
template<unsigned int BlockSize>
void configure_parser(cli::Parser& parser)
{
    // Default parameters.
    const constexpr unsigned int width      = 4096;
    const constexpr unsigned int height     = 4096;
    const constexpr unsigned int iterations = 10;
    // const constexpr bool         print      = false;

    parser.set_optional<unsigned int>("x", "width", width, "Width of the input grid");
    parser.set_optional<unsigned int>("y", "height", height, "Height of the input grid");
    parser.set_optional<unsigned int>("i",
                                      "iterations",
                                      iterations,
                                      "Number of times the algorithm is executed.");
    //parser.set_optional<bool>("p", "print", print, "Enables printing the convoluted grid");
}

int main(int argc, char* argv[])
{
    // Number of threads in each kernel block dimension.
    const constexpr unsigned int block_size = 32;
    const constexpr unsigned int mask_width = 5;


    // Get number of nodes and iterations from the command line, if provided.
    unsigned int width = 4096;
    unsigned int height = 4096;
    unsigned int iterations;

    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <iterations>" << std::endl;
        return 1; // Error exit code
    }

    // Parse the first argument as the number of steps (log2length)
    width = std::stoi(argv[1]);

    // Parse the second argument as the sort order (inc/dec)
    height = std::stoi(argv[2]);

    iterations = 1;

    // Check values provided.
    if(width < 1)
    {
        std::cout << "Width  must be at least 1. (provided " << width << " )" << std::endl;
        return 1;
    }
    if(height < 1)
    {
        std::cout << "Height  must be at least 1. (provided " << height << " )" << std::endl;
        return 1;
    }
    if(iterations < 1)
    {
        std::cout << "Iterations  must be at least 1. (provided " << iterations << " )"
                  << std::endl;
        return 1;
    }

    // Total number of elements and bytes of the input grid.
    const unsigned int size       = width * height;
    const unsigned int size_bytes = size * sizeof(float);

    const constexpr unsigned int mask_element_num = mask_width * mask_width;
    const constexpr unsigned int mask_size_bytes  = mask_element_num * sizeof(float);
    const constexpr unsigned int filter_radius    = mask_width / 2;

    const unsigned int padded_width            = width + filter_radius * 2;
    const unsigned int padded_height           = height + filter_radius * 2;
    const unsigned int input_size_padded       = padded_width * padded_height;
    const unsigned int input_size_padded_bytes = input_size_padded * sizeof(float);

    auto mask = convolution_filter_5x5;

    // Allocate unified host/device memory for input grid and output grid.
    float* input_grid_padded;
    float* output_grid;
    HIP_CHECK(hipMallocManaged(&input_grid_padded, input_size_padded_bytes));
    HIP_CHECK(hipMallocManaged(&output_grid, size_bytes));

    // Copy mask to constant memory.
    HIP_CHECK(hipMemcpyToSymbol(d_mask, mask.data(), mask_size_bytes));

    // Fill input grid with random data.
    std::mt19937 rng(0); // Seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate(input_grid_padded, input_grid_padded + input_size_padded, [&]() { return dist(rng); });

    // Define grid and block dimensions.
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    // Run the convolution kernel multiple times and measure performance.
    for(unsigned int i = 0; i < iterations; ++i)
    {
        hipLaunchKernelGGL(convolution<mask_width>, gridDim, blockDim, 0, 0, input_grid_padded, output_grid, make_uint2(width, height));
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

#ifdef DEBUG_CHECK
    // Verify the results with the reference implementation.
    std::vector<float> expected_output_grid(size);
    convolution_reference(expected_output_grid, std::vector<float>(input_grid_padded, input_grid_padded + input_size_padded), mask, height, width, mask_width);

    // Convert raw pointers to std::vector for comparison
    std::vector<float> output_grid_vector(output_grid, output_grid + size);

    // --- Compare Results ---
    bool is_equal = std::equal(output_grid_vector.begin(), output_grid_vector.end(), expected_output_grid.begin(),
                            [](float a, float b) { return std::fabs(a - b) < 1e-5; });

    if (is_equal) {
        std::cout << "The outputs are the same. Multithreaded version is correct." << std::endl;
    } else {
        std::cout << "The outputs are different. There may be an issue with the multithreaded version." << std::endl;
    }
#endif

    // Free memory.
    HIP_CHECK(hipFree(input_grid_padded));
    HIP_CHECK(hipFree(output_grid));

    return 0;
}
