#include <iostream>
#include <array>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include <pthread.h>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <cstdlib>  // for std::atoi

//#define DEBUG_CHECK

const unsigned int rows_per_batch = 1;  // Number of rows each thread processes at a time
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned int current_row = 0;  // Shared variable to track the current row being processed

// Define the 5x5 convolution filter
const constexpr std::array<float, 5 * 5> convolution_filter_5x5 = {1.0f,  3.0f, 0.0f,  -2.0f, -0.0f,
                                                                   1.0f,  4.0f, 0.0f,  -8.0f, -4.0f,
                                                                   2.0f,  7.0f, 0.0f, -12.0f, -0.0f,
                                                                   2.0f,  3.0f, 1.5f,  -8.0f, -4.0f,
                                                                   0.0f,  1.0f, 0.0f,  -2.0f, -0.0f};

// Thread data structure for passing to pthreads
struct ThreadData {
    const float* input;
    float* output;
    const float* mask;
    unsigned int width;
    unsigned int padded_width;
    unsigned int mask_width;
    unsigned int height;
    unsigned int thread_id;
    unsigned int rows_processed = 0;  // Counter to track number of rows processed
};

// Function for performing convolution, to be run by threads
void* convolution_thread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    const unsigned int width = data->width;
    const unsigned int padded_width = data->padded_width;
    const unsigned int mask_width = data->mask_width;
    const float* input = data->input;
    float* output = data->output;
    const float* mask = data->mask;
    const unsigned int height = data->height;

    while (true) {
        pthread_mutex_lock(&mutex);
        unsigned int start_row = current_row;
        current_row += rows_per_batch;
        pthread_mutex_unlock(&mutex);

        if (start_row >= height) {
            break;  // No more work to do
        }

        unsigned int end_row = std::min(start_row + rows_per_batch, height);

        for (unsigned int y = start_row; y < end_row; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y) {
                    for (unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x) {
                        unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                        unsigned int input_index = (y + mask_index_y) * padded_width + (x + mask_index_x);
                        sum += input[input_index] * mask[mask_index];
                    }
                }
                output[(y * width + x)] = sum;
            }
        }
        
#ifdef DEBUG_CHECK
        // Increment row count for the thread
        data->rows_processed += (end_row - start_row);
#endif
    }

    return nullptr;
}

// Single-threaded convolution function for comparison
void convolution_single_threaded(const float* input, float* output, const float* mask,
                                 unsigned int width, unsigned int padded_width, unsigned int mask_width, unsigned int height) {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y) {
                for (unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x) {
                    unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                    unsigned int input_index = (y + mask_index_y) * padded_width + (x + mask_index_x);
                    sum += input[input_index] * mask[mask_index];
                }
            }
            output[(y * width + x)] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    // Default values
    unsigned int width = 4096;
    unsigned int height = 4096;
    unsigned int num_threads = 8;

    // Check if the user provided arguments
    if (argc > 1) {
        width = std::atoi(argv[1]);  // First argument: width
    }
    if (argc > 2) {
        height = std::atoi(argv[2]);  // Second argument: height
    }
    if (argc > 3) {
        num_threads = std::atoi(argv[3]);  // Third argument: number of threads
    }

    const unsigned int mask_width = 5;
    const unsigned int filter_radius = mask_width / 2;
    const unsigned int padded_width = width + filter_radius * 2;
    const unsigned int padded_height = height + filter_radius * 2;
    const unsigned int size = width * height;
    const unsigned int input_size_padded = padded_width * padded_height;

    std::vector<float> input_grid(size);
    std::vector<float> output_grid(size);
    std::vector<float> output_grid_single_threaded(size);
    std::vector<float> input_grid_padded(input_size_padded, 0);

    // Generate random input
    std::mt19937 mersenne_engine{0};
    std::uniform_real_distribution<float> distribution{0, 256};
    auto rnd = std::bind(distribution, mersenne_engine);  // Use std::bind for random number generation
    std::generate(input_grid.begin(), input_grid.end(), rnd);

    // Copy input grid to padded input grid
    auto input_grid_row_begin = input_grid.begin();
    auto padded_input_grid_row_begin = input_grid_padded.begin() + filter_radius * padded_width + filter_radius;
    for (unsigned int i = 0; i < height; i++) {
        std::copy(input_grid_row_begin, input_grid_row_begin + width, padded_input_grid_row_begin);
        padded_input_grid_row_begin += padded_width;
        input_grid_row_begin += width;
    }

    // --- Multithreaded (Pthreads) Convolution ---
    ThreadData thread_data[num_threads];
    pthread_t threads[num_threads];

    for (unsigned int i = 0; i < num_threads; ++i) {
        thread_data[i].input = input_grid_padded.data();
        thread_data[i].output = output_grid.data();
        thread_data[i].mask = convolution_filter_5x5.data();
        thread_data[i].width = width;
        thread_data[i].padded_width = padded_width;
        thread_data[i].mask_width = mask_width;
        thread_data[i].height = height;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], nullptr, convolution_thread, &thread_data[i]);
    }

    // Join threads
    for (unsigned int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

#ifdef DEBUG_CHECK
    // --- Single-threaded Convolution ---
    convolution_single_threaded(input_grid_padded.data(), output_grid_single_threaded.data(),
                                convolution_filter_5x5.data(), width, padded_width, mask_width, height);

    // --- Compare Results ---
    bool is_equal = std::equal(output_grid.begin(), output_grid.end(), output_grid_single_threaded.begin(),
                               [](float a, float b) { return std::fabs(a - b) < 1e-5; });

    if (is_equal) {
        std::cout << "The outputs are the same. Multithreaded version is correct." << std::endl;
    } else {
        std::cout << "The outputs are different. There may be an issue with the multithreaded version." << std::endl;
    }

    int total_rows = 0;
    // Print number of rows processed by each thread
    for (unsigned int i = 0; i < num_threads; ++i) {
        total_rows = total_rows + thread_data[i].rows_processed;
        std::cout << "Thread " << i << " processed " << thread_data[i].rows_processed << " rows." << std::endl;
    }
    std::cout << "Total rows processed: " << total_rows << std::endl;
#endif

    return 0;
}
